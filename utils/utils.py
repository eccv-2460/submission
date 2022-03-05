import os
import time
import yaml
from pathlib import Path
import torch
import torch.distributed as dist


path_list = ['model', 'result', 'log']


def init_path():
    Path('./checkpoint').mkdir(parents=True, exist_ok=True)
    for path in path_list:
        Path('./checkpoint', path).mkdir(parents=True, exist_ok=True)


def check_path(save_path):
    path_dict = {}

    if save_path == '':
        save_path = time.strftime('%Y-%m-%d-%H-%M')
    for path in path_list:
        Path('./checkpoint', path, save_path).mkdir(parents=True, exist_ok=True)
        path_dict[path + '_path'] = str(Path('./checkpoint', path, save_path))

    return path_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class AverageMeter:
    def __init__(self, name='Average'):
        self.name = name
        self.sum = 0
        self.num = 0
        self.now = 0

    def update(self, data):
        self.sum += data
        self.num += 1
        self.now = data

    def get_now(self):
        return self.now

    def get_average(self, clean=False):
        res = self.sum / self.num
        if clean:
            self.clean()
        return res

    def get_name(self):
        return self.name

    def clean(self):
        self.sum = 0
        self.num = 0


def update_from_config_base(args, config):
    with open(config, 'r', encoding='utf-8') as file:
        conf = yaml.load(file.read(), Loader=yaml.FullLoader)
        for key in conf.keys():
            if conf[key]:
                setattr(args, key, conf[key])
    return args


def update_from_config(args, config):
    args = update_from_config_base(args, './config/base.yaml')
    if config:
        args = update_from_config_base(args, config)
    return args
