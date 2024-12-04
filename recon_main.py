import torch
import torch.nn as nn
import numpy as np
import os, sys
from pathlib import Path

from PointGeneration.ds_shape_color_generation import EEGTo3DDiffusionModel

from contextlib import nullcontext
from accelerate import Accelerator
from train_utils.parse import parse_args
from train_utils import training_utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datetime
from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from eeg_data_process.clip_loss import ClipLoss
import open3d as o3d
from accelerate import DistributedDataParallelKwargs
import time
###train: python recon_main.py --generation_type 'shape' --sub 'sub25' --in_channels 1027 --pretrain_model retri_color_shape_11-05_11-33_VideoImageEEGClassifyColor3_color_video_fea_time_len1
###test: python recon_main.py --generation_type 'shape' --task 'sample' --sub 'sub11' --in_channels 1027 --checkpoint_resume 
### '/home/bingxing2/ailab/ailab_share/scxlab0036/ai4neuro-gzq21/model/point_generate/shape/sub11/2024-10-31--00-10-05/checkpoint-40000.pth'

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
    if args.generation_type == 'shape':
        model = EEGTo3DDiffusionModel(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, sub=args.sub, generate_type=args.generation_type,
            retri_pretrain_model=f"{args.data_path}/model/retraival/{args.sub}/{args.pretrain_model}/",point_cloud_model_embed_dim=args.point_cloud_model_embed_dim, in_channels=args.in_channels,out_channels=args.out_channels)
    else:
        model = EEGTo3DDiffusionModel(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, sub=args.sub, generate_type=args.generation_type,
            point_cloud_model_embed_dim=args.point_cloud_model_embed_dim, in_channels=args.in_channels, out_channels=args.out_channels+3)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')
    optimizer = training_utils.get_optimizer(args, model, accelerator)
    scheduler = training_utils.get_scheduler(args, optimizer)
    
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(args, model, optimizer, scheduler)

    train_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=True, aug_data=True)
    test_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=False, point_path=args.ply_point_path)
    point_features_train_all = train_set.color_point_features[:, 0].float()
    video_features_train_all = train_set.color_video_features[:, 0].float()
    point_features_test_all = test_set.color_point_features[:, 0].float()
    video_features_test_all = test_set.color_video_features[:, 0].float()
    
    dataloader_test = DataLoader(dataset=test_set, batch_size=48, num_workers=args.num_workers, shuffle=False)
    dataloader_train = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    total_batch_size = args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    model, optimizer, scheduler, dataloader_train, dataloader_test = accelerator.prepare(model, optimizer, scheduler, dataloader_train, dataloader_test)
    
    if args.task == 'sample':
        if args.checkpoint_resume == '':
            print('please input checkpoint path!')
            return
        model_save_path = args.checkpoint_resume[:-4] + '/'
        print(model_save_path)
        os.makedirs(model_save_path, exist_ok=True)
        video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
        print(video_acc)
        # import pdb;pdb.set_trace()
        visualize(args, model, dataloader_test, accelerator, train_state.step, point_features_test_all, video_features_test_all, model_save_path, infer_steps=1000, num=5)
        return 
    video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
    print(video_acc)
    video_acc = retrieval_test(args, model, dataloader_train, point_features_train_all, video_features_train_all)
    print(video_acc)
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
    model_save_path = args.data_path + args.model_save_path + args.generation_type + '/' + args.sub + '/' + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)
    log_info_txt = open(model_save_path + '/log.txt', 'w')
    log_info_txt.write(ss)
    log_info_txt.flush()
    print(ss)
    print(get_parameter_number(model))
    
    best_video_acc = 0
    while True:
        video_acc_count_all, total_all = 0, 0
        for i, batch in enumerate(dataloader_train):
            model.train()
            if args.generation_type == 'shape':
                point_c = None
                pc = batch['point_cloud'].float()[:, :, :3]
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['cls_label']
            elif args.generation_type == 'color':
                point_c = batch['point_cloud'].float()[:, :, :3]
                pc = batch['point_cloud'].float()[:, :, 3:]
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['color_label']
            else:
                return
            fea_list = {'point_features': point_features, 'video_features': video_features,
            'point_features_all':point_features_train_all, 'video_features_all':video_features_train_all}
            with accelerator.accumulate(model):
                loss_dm = model(pc, eeg_data, eeg_data2, mode='train', shape_c=point_c, fea_list=fea_list, labels=labels)
                loss = loss_dm

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
                    if video_acc > best_video_acc:
                        best_video_acc = video_acc
                    step_s = str(train_state.step).zfill(7)
                    ss = '\nstep: ' + step_s + ',  '
                    ss += f'lr: {optimizer.param_groups[0]["lr"]}, train_loss:{loss_value} DM:{float(loss_dm)}\n'
                    ss += f'test video_acc:{video_acc:.4f}\n'
                    ss += f'test best video_acc:{best_video_acc:.4f}\n'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                    print(ss)

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
                if train_state.step >= args.max_steps:
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    ss = f'Ending training at: {datetime.datetime.now()}\n'
                    ss += f'Final train state: {train_state}'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                    log_info_txt.close()
                    return
                if train_state.step >= 100500:
                    log_info_txt.flush()
                    log_info_txt.close()
                    return

def visualize(args, model, dataloader_test, accelerator, epoch_step, point_features_test_all, video_features_test_all, model_save_path, infer_steps=200, num=1):
    model.eval()
    for num_index in range(0, num):
        time_b = time.time()
        video_acc_count_all, total_all = 0, 0
        for batch_idx, batch in enumerate(dataloader_test):
            if args.generation_type == 'shape':
                point_c = None
                pc = batch['point_cloud'].float()[:, :, :3]
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['cls_label']
            elif args.generation_type == 'color':
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
            output, (video_acc_count, total, acc_list) = model(pc, eeg_data, eeg_data2, mode='sample', shape_c=point_c, fea_list=fea_list, labels=labels,
                                                                    return_sample_every_n_steps=-1, num_inference_steps=infer_steps, disable_tqdm=(not accelerator.is_main_process))
            video_acc_count_all += video_acc_count
            total_all += total
            for ii in range(0, output.shape[0]):
                point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
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
                name = batch['name'][ii]
                if acc_list[ii]:
                    print(name * 10)
                o3d.io.write_point_cloud(f'{model_save_path}/{epoch_step}-{name}-{num_index}.ply', pcd)
        time_e = time.time()
        print(time_e - time_b)
        print(f'step:{epoch_step}, testset, video acc:{(video_acc_count_all / total_all):.4f}')
    model.train()

def retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all):
    video_acc_count_all, total_all = 0, 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader_test):
        if args.generation_type == 'shape':
            point_c = None
            pc = batch['point_cloud'].float()[:, :, :3]
            eeg_data = batch['eeg_data'].float()
            eeg_data2 = batch['eeg_data2'].float()
            point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
            labels = batch['cls_label']
        elif args.generation_type == 'color':
            point_c = batch['point_cloud'].float()[:, :, :3]
            pc = batch['point_cloud'].float()[:, :, 3:]
            eeg_data = batch['eeg_data'].float()
            eeg_data2 = batch['eeg_data2'].float()
            point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
            labels = batch['color_label']
        else:
            return
        fea_list = {'point_features': point_features, 'video_features': video_features,
        'point_features_all':point_features_test_all, 'video_features_all':video_features_test_all}
        (video_acc_count, total, acc_list) = model(pc, eeg_data, eeg_data2, mode='test_retrieval', shape_c=point_c, fea_list=fea_list, labels=labels)
        video_acc_count_all += video_acc_count
        total_all += total
    model.train()
    return video_acc_count_all / total_all

if __name__ == '__main__':
    main()

