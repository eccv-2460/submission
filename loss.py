import numpy as np
import torch


bce = torch.nn.BCELoss()


def compute_mse_loss(pred, target, trimap):
    error_map = pred - target
    loss = np.sum((error_map ** 2) * (trimap == 1)) / (np.sum(trimap == 1) + 1e-8)

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs(pred - target)
    loss = np.sum(error_map * (trimap == 1))

    return loss / 1000, np.sum(trimap == 1) / 1000


def gradient_loss(result, gt):
    def image_gradient(image):
        batch_size, depth, height, width = image.shape
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        dx = image[:, :, :, 1:] - image[:, :, :, :-1]

        dy = torch.cat([dy, torch.zeros((batch_size, depth, 1, width)).cuda()], 2)
        dx = torch.cat([dx, torch.zeros((batch_size, depth, height, 1)).cuda()], 3)

        return dy, dx

    epsilon = 1e-07
    gt_dy, gt_dx = image_gradient(gt)
    result_dy, result_dx = image_gradient(result)

    dy_loss_square = (gt_dy - result_dy) ** 2
    dx_loss_square = (gt_dx - result_dx) ** 2

    dy_loss = (dy_loss_square + epsilon) ** 0.5
    dx_loss = (dx_loss_square + epsilon) ** 0.5
    return dy_loss.mean() + dx_loss.mean()


def regression_loss(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel
    :return:
    """
    if weight is None:
        if loss_type == 'l1':
            return torch.nn.functional.l1_loss(logit, target)
        elif loss_type == 'l2':
            return torch.nn.functional.mse_loss(logit, target)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
    else:
        if loss_type == 'l1':
            return torch.nn.functional.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        elif loss_type == 'l2':
            return torch.nn.functional.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


def smooth_l1(logit, target, weight):
    loss = torch.sqrt((logit * weight - target * weight) ** 2 + 1e-6)
    loss = torch.sum(loss) / (torch.sum(weight) + 1e-8)
    return loss


def grad_loss(logit, target, grad_filter, loss_type='l1', weight=None):
    """ pass """
    grad_logit = torch.nn.functional.conv2d(logit, weight=grad_filter, padding=1)
    grad_target = torch.nn.functional.conv2d(target, weight=grad_filter, padding=1)
    grad_logit = torch.sqrt((grad_logit * grad_logit).sum(dim=1, keepdim=True) + 1e-8)
    grad_target = torch.sqrt((grad_target * grad_target).sum(dim=1, keepdim=True) + 1e-8)

    return regression_loss(grad_logit, grad_target, loss_type=loss_type, weight=weight)


def gabor_loss(logit, target, gabor_filter, loss_type='l2', weight=None):
    """ pass """
    gabor_logit = torch.nn.functional.conv2d(logit, weight=gabor_filter, padding=2)
    gabor_target = torch.nn.functional.conv2d(target, weight=gabor_filter, padding=2)

    return regression_loss(gabor_logit, gabor_target, loss_type=loss_type, weight=weight)


def composition_loss(alpha, fg, bg, image, weight, loss_type='l1'):
    """
    Alpha composition loss
    """
    merged = fg * alpha + bg * (1 - alpha)
    return regression_loss(merged, image, loss_type=loss_type, weight=weight)


def bce_loss(predict, target):
    return bce(predict, target)


def fm_loss(predict, target):
    TP = (predict * target).sum(dim=[2, 3])
    FP = (predict * (1 - target)).sum(dim=[2, 3])
    FN = ((1 - predict) * target).sum(dim=[2, 3])

    p = TP / (TP + FP + 1e-7)
    r = TP / (TP + FN + 1e-7)

    fm = 1.3 * p * r / (0.3 * p + r + 1e-7)
    fm = fm.clamp(min=1e-7, max=1 - 1e-7)

    return 1 - fm.mean()
