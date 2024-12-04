"""
Misc functions, including distributed helpers, mostly from torchvision
"""
import math
import time
import datetime
import random
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Callable, Optional
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from accelerate import Accelerator
from omegaconf import DictConfig
from dataclasses import dataclass, field



@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_val: Optional[float] = None


def get_optimizer(args, model: torch.nn.Module, accelerator: Accelerator) -> torch.optim.Optimizer:
    """Gets optimizer from config"""
    
    lr = args.lr
    print('lr = {lr} (absolute learning rate)'.format(lr=lr))

    # Get optimizer parameters, excluding certain parameters from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    Optimizer: torch.optim.Optimizer = getattr(torch.optim, 'AdamW')
    optimizer = Optimizer(parameters, lr=lr, betas=(0.95, 0.999))

    return optimizer


def get_scheduler(args, optimizer: torch.optim.Optimizer) -> Callable:
    """Gets scheduler from config"""
    from transformers import get_scheduler
    if args.scheduler_type == 'cosine':
        scheduler = get_scheduler(optimizer=optimizer, name='cosine', num_warmup_steps=2000, num_training_steps=args.max_steps,)
    elif args.scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer=optimizer, name='linear', num_warmup_steps=0, num_training_steps=args.max_steps,)
    else:
        return
    return scheduler


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, device='cuda'):
        """
        Warning: does not synchronize the deque!
        """
        if not using_distributed():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value) if len(self.deque) > 0 else ""


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        n = kwargs.pop('n', 1)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, device='cuda'):
        for meter in self.meters.values():
            meter.synchronize_between_processes(device=device)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{}  Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def resume_from_checkpoint(args, model, optimizer=None, scheduler=None, model_ema=None):
    
    # Check if resuming training from a checkpoint
    if args.checkpoint_resume == '':
        print('Starting training from scratch')
        return TrainState()

    # If resuming, load model state dict
    print(f'Loading checkpoint ({datetime.datetime.now()})')
    checkpoint = torch.load(args.checkpoint_resume, map_location='cpu')
    if 'model' in checkpoint:
        state_dict, key = checkpoint['model'], 'model'
    else:
        state_dict, key = checkpoint, 'N/A'
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print('Removed "module." from checkpoint state dict')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded model checkpoint key {key} from {args.checkpoint_resume}')
    if len(missing_keys):
        print(f' - Missing_keys: {missing_keys}')
    if len(unexpected_keys):
        print(f' - Unexpected_keys: {unexpected_keys}')

    train_state = TrainState()
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch, step, best_val = checkpoint['epoch'] + 1, checkpoint['step'], checkpoint['best_val']
    train_state = TrainState(epoch=epoch, step=step, best_val=best_val)
    print(f'Finished loading checkpoint ({datetime.datetime.now()})')
    return train_state


def setup_distributed_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    from rich import print as __richprint__
    builtin_print = __richprint__  # __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def using_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if using_distributed() else 0


def set_seed(seed):
    rank = get_rank()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if using_distributed():
        print(f'Seeding node {rank} with seed {seed}', force=True)
    else:
        print(f'Seeding node {rank} with seed {seed}')


def compute_grad_norm(parameters):
    # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2).item()
    total_norm = 0
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
