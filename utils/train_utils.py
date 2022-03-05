import torch
from utils.utils import AverageMeter
from tqdm import tqdm
from loss import compute_mse_loss, compute_sad_loss
from loss import regression_loss, smooth_l1, gradient_loss
from loss import bce_loss, fm_loss
import cv2
import os
from pathlib import Path
import numpy as np


def generate_patch_ground_truth(trimap):
    b, _, h, w = trimap.shape
    trimap = trimap.reshape(b, 1, h // 4, 4, w // 4, 4).permute(0, 1, 3, 5, 2, 4)
    trimap = trimap.reshape(b, 1, 4 * 4, h // 4, w // 4).sum(dim=2)
    gt = torch.ones_like(trimap) * 0.5
    gt[trimap == 0] = 0
    gt[trimap == 32] = 1
    return gt


def generate_cls_ground_truth_alpha(alpha):
    full_gt = torch.zeros_like(alpha).long()
    full_gt[alpha > 0.95] = 1
    t1, t2 = alpha >= 0.05, alpha <= 0.95
    full_gt[t1 & t2] = 2
    return full_gt


def generate_cls_ground_truth_trimap(trimap):
    full_gt = torch.zeros_like(trimap).long()
    full_gt[trimap == 0.5] = 2
    full_gt[trimap == 1] = 1
    return full_gt


def generate_fg_ground_truth(alpha):
    seg = torch.where(alpha > 0.95, torch.ones_like(alpha), torch.zeros_like(alpha))
    return seg


def generate_image_result(output):
    output_softmax = torch.nn.functional.softmax(output, 1)
    bg, fg, un = torch.split(output_softmax, 1, 1)
    bg, fg, un = bg.detach().cpu().numpy().astype(np.float32), fg.detach().cpu().numpy().astype(np.float32), \
                 un.detach().cpu().numpy().astype(np.float32)
    output_image = np.zeros_like(bg)
    output_image = output_image + un + fg * 2
    return bg, fg, un, output_image


def train_one_epoch(data_loader, model, optimizer, device, epoch, loss_scaler, fp32, writer, k, args):
    # training code will be released if this work is accepted
    pass

@torch.no_grad()
def evaluate(data_loader, model, device, epoch, path_dict):
    model.eval()

    Path(path_dict['result_path'], str(epoch)).mkdir(parents=True, exist_ok=True)

    mse = AverageMeter('mse')
    sad = AverageMeter('sad')

    for image, alpha, trimap, filename in tqdm(data_loader):
        image, trimap = image.to(device, non_blocking=True), trimap.to(device, non_blocking=True)
        # input_image = torch.cat([image, trimap], dim=1)

        with torch.cuda.amp.autocast():
            output = model(image)
            output_alpha = output['reg']
        # output[trimap == 2] = 1
        # output[trimap == 0] = 0

        batch_size = output_alpha.shape[0]
        output_alpha = output_alpha.detach().cpu().numpy().astype(np.float32)
        alpha = alpha.data.cpu().numpy()
        trimap = trimap.detach().cpu().numpy()
        for i in range(batch_size):
            mse.update(compute_mse_loss(output_alpha[i], alpha[i], trimap[i]))
            sad.update(compute_sad_loss(output_alpha[i], alpha[i], trimap[i])[0])
            cv2.imwrite(os.path.join(path_dict['result_path'], str(epoch), filename[i]), output_alpha[i][0] * 255)

    return {'mse': mse.get_average(), 'sad': sad.get_average()}


class log_best:
    def __init__(self, init, epoch, func):
        self.best = init
        self.epoch = epoch
        self.compare = func

    def update(self, value, epoch):
        if self.compare(value, self.best):
            self.best = value
            self.epoch = epoch
