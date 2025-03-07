import torch
import torch.nn as nn
import numpy as np
import os, sys
from pathlib import Path

from PointGeneration.eeg_add_color import EEGTo3DDiffusionModel

from contextlib import nullcontext
from accelerate import Accelerator
import torch.nn.functional as F
from train_utils.parse import parse_args
from train_utils import training_utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datetime
from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from eeg_data_process.clip_loss import ClipLoss
import open3d as o3d
from accelerate import DistributedDataParallelKwargs
import math

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False, gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    training_utils.setup_distributed_print(accelerator.is_main_process)
    print(f'Current working directory: {os.getcwd()}')
    print(args)
    training_utils.set_seed(args.seed)
    model = EEGTo3DDiffusionModel(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, sub=args.sub,
            pretrain_model=f"{args.data_path}/model/retraival/{args.sub}/{args.pretrain_model}/", point_cloud_model_embed_dim=args.point_cloud_model_embed_dim, in_channels=args.in_channels, out_channels=args.out_channels, model_type=args.model_type)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')
    optimizer = training_utils.get_optimizer(args, model, accelerator)
    scheduler = training_utils.get_scheduler(args, optimizer)
    
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(args, model, optimizer, scheduler)

    # test_set = OursEEGPoint(data_path, obj_n=10)
    # train_set = OursEEGPoint(data_path, obj_n=10)
    train_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=True, aug_data=True)
    test_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=False, point_path=args.ply_point_path)
    point_features_train_all = train_set.color_point_features[:, 0].float()
    video_features_train_all = train_set.color_video_features[:, 0].float()
    point_features_test_all = test_set.color_point_features[:, 0].float()
    video_features_test_all = test_set.color_video_features[:, 0].float()

    # test_set = CLIPPoint(args.data_path, obj_n=8, train=False)
    # train_set = CLIPPoint(args.data_path, obj_n=8, train=True)
    dataloader_test = DataLoader(dataset=test_set, batch_size=72, num_workers=args.num_workers, shuffle=False)
    dataloader_train = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    total_batch_size = args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    model, optimizer, scheduler, dataloader_train, dataloader_test = accelerator.prepare(model, optimizer, scheduler, dataloader_train, dataloader_test)
    
    if args.task == 'sample':
        if args.checkpoint_resume == '':
            print('please input checkpoint path!')
            return
        model_save_path = args.checkpoint_resume[:-4] + '/'
        print(model_save_path)
        video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
        print(video_acc)
        os.makedirs(model_save_path, exist_ok=True)
        visualize(args, model, dataloader_test, accelerator, train_state.step, point_features_test_all, video_features_test_all, model_save_path, infer_steps=1000, num=1)
        return 
    # import pdb;pdb.set_trace()
    test_video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
    # import pdb;pdb.set_trace()
    train_acc = retrieval_test(args, model, dataloader_train, point_features_train_all, video_features_train_all)
    print(f"test_video_acc: {test_video_acc:.4f}, train_acc: {train_acc:.4f}")
    ss = f'\n***** Starting training at {datetime.datetime.now()} *****\n'
    ss += f'    Dataset train size: {len(dataloader_train.dataset):_}\n'
    ss += f'    Dataset val size: {len(dataloader_train.dataset):_}\n'
    ss += f'    Dataloader train size: {len(dataloader_train):_}\n'
    ss += f'    Batch size per device = {args.batch_size}\n'
    ss += f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}\n'
    ss += f'    Gradient Accumulation steps = {args.gradient_accumulation_steps}\n'
    ss += f'    Max training steps = {args.max_steps}\n'
    ss += f'    Training state = {train_state}\n'
    # if accelerator.is_main_process:
    model_save_path = args.data_path + args.model_save_path + args.generation_type + '/' + args.sub + '/' + args.model_type + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)
    log_info_txt = open(model_save_path + '/log.txt', 'w')
    log_info_txt.write(ss)
    log_info_txt.flush()
    print(ss)
    print(get_parameter_number(model))
    min_test_loss = 100
    while True:
        for i, batch in enumerate(dataloader_train):
            model.train()
            if args.generation_type == 'color':
                point_c = batch['point_cloud'].float()[:, :, :3]
                pc = batch['point_cloud'].float()[:, :, 3:]
                # c1, c2 = batch['color_video_fea'].float(), batch['color_point_fea'].float()
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['cls_label']
            else:
                return
            fea_list = {'point_features': point_features, 'video_features': video_features,
            'point_features_all':point_features_train_all, 'video_features_all':video_features_train_all}
            # c1, c2 = torch.rand(pc.shape[0], 1024).to(pc.device).float(), torch.rand(pc.shape[0], 768).to(pc.device).float()
            with accelerator.accumulate(model):
                # import pdb;pdb.set_trace()
                loss = model(pc, eeg_data, eeg_data2, mode='train', shape_c=point_c, noise_std=0.02)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1
                loss_value = loss.item()
            if accelerator.sync_gradients:
                if accelerator.is_main_process and train_state.step % args.log_step_freq == 0:
                    video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
                    step_s = str(train_state.step).zfill(7)
                    test_loss = get_test_loss(args, model, dataloader_test)
                    ss = '\nstep: ' + step_s + ',  '
                    ss += f'lr: {optimizer.param_groups[0]["lr"]}, loss:{float(loss):.5f}, test_loss:{test_loss:.4f} video_acc: {video_acc:.4f}\n'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                    print(ss)
                    if test_loss < min_test_loss:
                        min_test_loss = test_loss
                        checkpoint_dict = {
                            'model': accelerator.unwrap_model(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': train_state.epoch,
                            'step': train_state.step,
                            'best_val': train_state.best_val
                        }
                        checkpoint_path = f'{model_save_path}/checkpoint-best.pth'
                        accelerator.save(checkpoint_dict, checkpoint_path)
                        print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')
                        
                    
                # Save a checkpoint
                if accelerator.is_main_process and (train_state.step % args.checkpoint_freq == 0 or train_state.step == 1):
                    
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val
                    }
                    checkpoint_path = f'{model_save_path}/checkpoint-{train_state.step}.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')
                
                # if accelerator.is_main_process and (train_state.step % args.test_freq == 0 or train_state.step == 1):
                #     visualize(args, model, dataloader_test, accelerator, train_state.step, point_features_test_all, video_features_test_all, model_save_path)
                    
                # End training after the desired number of steps/epochs
                if train_state.step >= args.max_steps:
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    ss = f'Ending training at: {datetime.datetime.now()}\n'
                    ss += f'Final train state: {train_state}'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                    log_info_txt.close()
                    return

def visualize(args, model, dataloader_test, accelerator, epoch_step, point_features_test_all, video_features_test_all, model_save_path, infer_steps=200, num=1):
    model.eval()
    if args.ply_point_path != '':
        point_path = args.ply_point_path.split('/')[-2].split('-')[-1]
    else:
        point_path = 'gt'
    for num_index in range(0, num):
        color_error_sum, all_num = 0, 0
        for batch_idx, batch in enumerate(dataloader_test):
            if args.generation_type == 'color':
                point_c = batch['point_cloud'].float()[:, :, :3]
                pc = batch['point_cloud'].float()[:, :, 3:]
                # c1, c2 = batch['color_video_fea'].float(), batch['color_point_fea'].float()
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['cls_label']
            else:
                return
            
            output = model(pc, eeg_data, eeg_data2, mode='sample', shape_c=point_c)
            for ii in range(0, output.shape[0]):
                name = batch['name'][ii]
                # print(name, (pc[ii] - output[ii]).mean())
                color_error_sum += math.fabs(float((pc[ii] - output[ii]).mean()))
                all_num += 1
                point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                # print(point_pred.min(), point_pred.max(), float(pc[ii].min()), float(pc[ii].max()))
                pcd = o3d.geometry.PointCloud()
                if args.generation_type == 'shape':
                    pcd.points = o3d.utility.Vector3dVector(point_pred)
                elif args.generation_type == 'color':
                    pcd.points = o3d.utility.Vector3dVector(point_c[ii].detach().cpu().numpy())
                    point_pred = (point_pred + 1.0) / 2.0
                    # print(point_pred.min(), point_pred.max(), point_lbl.min(), point_lbl.max())
                    point_pred[point_pred <= 0] = 0
                    point_pred[point_pred >= 1] = 1.0
                    pcd.colors = o3d.utility.Vector3dVector(point_pred)
                o3d.io.write_point_cloud(f"{model_save_path}/{point_path}-{name}-{num_index}.ply", pcd)
        print('error mean:', color_error_sum / all_num)
    color_result_record = open('/home/bingxing2/ailab/ailab_share/scxlab0036/ai4neuro-gzq21/model/point_generate/color/record-best1.txt', 'a')
    sub = args.sub
    test_ch = args.checkpoint_resume.split('/')[-2][:2] + '-' + args.checkpoint_resume.split('/')[-1].split('-')[-1]
    mean_err = color_error_sum / all_num / 2
    ss = f"{sub}, {test_ch}, {point_path}\n"
    ss += f"{mean_err:.3f}\n"
    print(ss)
    color_result_record.write(ss)
    color_result_record.flush()
    color_result_record.close()
    model.train()

def retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all):
    video_acc_count_all, total_all = 0, 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader_test):
        if args.generation_type == 'color':
            point_c = batch['point_cloud'].float()[:, :, :3]
            pc = batch['point_cloud'].float()[:, :, 3:]
            # c1, c2 = batch['color_video_fea'].float(), batch['color_point_fea'].float()
            eeg_data = batch['eeg_data'].float()
            eeg_data2 = batch['eeg_data2'].float()
            point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
            labels = batch['color_label']
        else:
            return
        fea_list = {'point_features': point_features, 'video_features': video_features,
        'point_features_all':point_features_test_all, 'video_features_all':video_features_test_all}
        (video_acc_count, total, acc_list) = model(pc, eeg_data, eeg_data2, mode='test_retrieval', shape_c=point_c, fea_list=fea_list, labels=labels)
        # for ii in range(0, pc.shape[0]):
        #     name = batch['name'][ii]
        #     if acc_list[ii]:
        #         print(name, labels[ii])
        video_acc_count_all += video_acc_count
        total_all += total
    model.train()
    return video_acc_count_all / total_all

def get_test_loss(args, model, dataloader_test):
    model.eval()
    loss_test_all = 0.0
    for batch_idx, batch in enumerate(dataloader_test):
        if args.generation_type == 'color':
            point_c = batch['point_cloud'].float()[:, :, :3]
            pc = batch['point_cloud'].float()[:, :, 3:]
            # c1, c2 = batch['color_video_fea'].float(), batch['color_point_fea'].float()
            eeg_data = batch['eeg_data'].float()
            eeg_data2 = batch['eeg_data2'].float()
            point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
            labels = batch['cls_label']
        else:
            return
        
        output = model(pc, eeg_data, eeg_data2, mode='sample', shape_c=point_c)
        loss_test = F.mse_loss(output, pc)
        loss_test_all += float(loss_test)
    model.train()
    return loss_test_all / (batch_idx + 1)

if __name__ == '__main__':
    main()

