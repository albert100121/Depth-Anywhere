import os
import sys
import torch
import torch.nn as nn


#==========================
# Depth Prediction Metrics
#==========================
# From https://github.com/zhijieshen-bjtu/PanoFormer
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


# From https://github.com/zhijieshen-bjtu/PanoFormer
class Evaluator(object):

    def __init__(self, median_align=True):

        self.median_align = median_align
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/mre"] = AverageMeter()
        self.metrics["err/mae"] = AverageMeter()
        self.metrics["err/abs_"] = AverageMeter()
        self.metrics["err/abs_rel"] = AverageMeter()
        self.metrics["err/sq_rel"] = AverageMeter()
        self.metrics["err/rms"] = AverageMeter()
        self.metrics["err/log_rms"] = AverageMeter()
        self.metrics["err/log10"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.metrics["err/mre"].reset()
        self.metrics["err/mae"].reset()
        self.metrics["err/abs_"].reset()
        self.metrics["err/abs_rel"].reset()
        self.metrics["err/sq_rel"].reset()
        self.metrics["err/rms"].reset()
        self.metrics["err/log_rms"].reset()
        self.metrics["err/log10"].reset()
        self.metrics["acc/a1"].reset()
        self.metrics["acc/a2"].reset()
        self.metrics["acc/a3"].reset()

    def compute_eval_metrics(self, gt_depth, pred_depth, mask=None):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]

        mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align)

        self.metrics["err/mre"].update(mre, N)
        self.metrics["err/mae"].update(mae, N)
        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/mre"].avg)
        avg_metrics.append(self.metrics["err/mae"].avg)
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)

        print("\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 11).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("\n  " + ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log",
                                                      "log10", "a1", "a2", "a3"), file=f)
                print(("&  {: 8.5f} " * 11).format(*avg_metrics), file=f)


def affine_invariant(depth:torch.tensor, mask:torch.tensor=None):
    # Median, t(d)
    B, _, _ = mask.shape
    median = torch.zeros((B,1,1), dtype=depth.dtype, device=depth.device)
    # scale, s(d)
    scale = torch.zeros((B,1,1), dtype=depth.dtype, device=depth.device)
    
    for idx in range(B):
        if mask[idx].sum() > 0 and torch.abs(depth[idx]).sum() > 0:
            # median of empty array leads to nan
            median[idx] = torch.median(depth[idx][mask[idx]])
            sd = torch.abs(depth[idx] - median[idx])
            scale[idx] = torch.mean(sd[mask[idx]])
            if scale[idx] == 0:
                scale[idx] = float('inf')
        else:
            scale[idx] = float('inf')
    depth_hat = depth - median.view(B, 1, 1)
    depth_hat = depth_hat / scale.view(B, 1, 1)
    if torch.isnan(depth_hat).sum() + torch.isinf(depth_hat).sum() > 0:
        print("Found nan or inf in depth_hat in scale and shift calculation")

    return depth_hat


def affine_invariant_loss_V2(pred:torch.tensor, gt:torch.tensor, mask:torch.tensor=None):
    if mask is None:
        mask = gt > 0
    pred *= mask.float()
    gt *= mask.float()
    if len(mask.shape) == 4:
        mask = mask.squeeze(1)
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(gt.shape) == 4:
        gt = gt.squeeze(1)

    losses = dict()
    pred_hat = affine_invariant(pred, mask)
    gt_hat = affine_invariant(gt, mask)

    losses['loss'] = torch.mean(torch.abs(pred_hat[mask]- gt_hat[mask]))

    return losses


def affine_invariant_loss_V2_CUBE(outputs, inputs, gt_w=0.5, pseudo_w=0.5):
    gt = inputs["gt_depth"]
    mask = inputs["val_mask"]
    equi_batch = gt.shape[0]

    pred = outputs["pred_depth"][:equi_batch]
    if torch.isnan(pred).sum() + torch.isinf(pred).sum() > 0:
        print("Found nan or inf in pred of loss function")
        import pdb
        pdb.set_trace()

    # Supervised #
    if mask is None:
        mask = gt > 0

    pred *= mask.float()
    gt *= mask.float()
    if len(mask.shape) == 4:
        mask = mask.squeeze(1)
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(gt.shape) == 4:
        gt = gt.squeeze(1)

    losses = dict()
    pred_hat = affine_invariant(pred, mask)
    gt_hat = affine_invariant(gt, mask)

    losses['loss_equi'] = torch.mean(torch.abs(pred_hat[mask]- gt_hat[mask]))

    if "pseudo_depth" in inputs:
        gt_cube = inputs["pseudo_depth"]
        mask_cube = inputs["pseudo_mask"]
        # cube_batch = gt_cube.shape[0]
        pred_disp_cube = outputs["pred_depth_cube"]
        pred_disp_cube *= mask_cube.float()
        gt_cube *= mask_cube.float()
        if len(mask_cube.shape) == 4:
            mask_cube = mask_cube.squeeze(1)
        if len(pred_disp_cube.shape) == 4:
            pred_disp_cube = pred_disp_cube.squeeze(1)
        if len(gt_cube.shape) == 4:
            gt_cube = gt_cube.squeeze(1)

        pred_hat = affine_invariant(pred_disp_cube, mask_cube)
        gt_hat = affine_invariant(gt_cube, mask_cube)
        losses['loss_cube'] = torch.mean(torch.abs(pred_hat[mask_cube]- gt_hat[mask_cube]))
        losses['loss'] = gt_w * losses['loss_equi'] + pseudo_w *losses['loss_cube'] 
    else:
        losses['loss'] = losses["loss_equi"]
    return losses


### Depth-Anything ###
def compute_scale_and_shift(prediction:torch.tensor, target:torch.tensor, mask:torch.tensor):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, h=512, w=1024):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, inputs, outputs, interpolate:bool=True, return_interpolated:bool=False, gt_w: float=0.5, pseudo_w: float=0.5):

        return affine_invariant_loss_V2_CUBE(outputs, inputs, gt_w, pseudo_w)


def absRel(pred, gt, mask=None):
    return torch.abs(pred[mask] - gt[mask]) / gt[mask]


def delta(pred, gt, mask=None):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = torch.mean(thresh < 1.25)
    a2 = torch.mean(thresh < 1.25 ** 2)
    a3 = torch.mean(thresh < 1.25 ** 3)

    return a1, a2, a3


def compute_ssi_depth_metrics(gt, pred, mask=None):
    """Computation of metrics between predicted and ground truth depths
    """
    gt_depth = gt
    pred_depth = pred

    # clip range in min max
    pred_depth = torch.clamp(pred_depth, 0.1, 10)
    gt_depth = torch.clamp(gt_depth, 0.1, 10)

    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean error###################

    rmse = (gt_depth - pred_depth) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt_depth + 1) - torch.log10(pred_depth + 1)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt_depth - pred_depth))

    abs_rel = torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth)

    sq_rel = torch.mean((gt_depth - pred_depth) ** 2 / gt_depth)

    log10 = torch.mean(torch.abs(torch.log10(pred_depth / gt_depth)))

    mae = torch.mean(torch.abs(pred_depth - gt_depth))
    
    mre = torch.mean(torch.abs(pred_depth - gt_depth) / gt_depth)

    mse = torch.mean(((pred_depth - gt_depth)**2) / gt_depth)

    return mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


def compute_depth_metrics(gt, pred, mask=None, median_align=True, max_depth: float=10.0):
    """Computation of metrics between predicted and ground truth depths
    """
    
    if mask is None:
        mask = gt > 0
    
    gt_depth = gt[mask]
    pred_depth = pred[mask]

    if median_align:
        pred_depth *= torch.median(gt_depth) / torch.median(pred_depth)
    
    pred_depth = torch.clamp(pred_depth, 0.1, max_depth)
    gt_depth = torch.clamp(gt_depth, 0.1, max_depth)

    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean error###################

    rmse = (gt_depth - pred_depth) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt_depth + 1) - torch.log10(pred_depth + 1)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt_depth - pred_depth))

    abs_rel = torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth)

    sq_rel = torch.mean((gt_depth - pred_depth) ** 2 / gt_depth)

    log10 = torch.mean(torch.abs(torch.log10(pred_depth / gt_depth)))

    mae = torch.mean(torch.abs(pred_depth - gt_depth))
    
    mre = torch.mean(torch.abs(pred_depth - gt_depth) / gt_depth)

    return mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


class Affine_Inv_Evaluator(Evaluator):

    def __init__(self, median_align=True, crop: int=0, max_depth: float=10.0):
        super().__init__(median_align=median_align)
        self.crop = crop
        if self.crop > 0:
            print(f"EVAL with cropping top bottom {self.crop}")
        self.max_depth = max_depth

    def compute_affine_inv_eval_metrics(self, gt_depth, pred_depth, mask=None):
        """
        Computes metrics used to evaluate the model
        :gt_depth: depth in meters
        :pred_depth: disparity up to scale
        """
        # crop here
        if self.crop > 0:
            pred_depth = pred_depth[..., self.crop:-self.crop, :]
            gt_depth = gt_depth[..., self.crop:-self.crop, :]
            mask = mask[..., self.crop:-self.crop, :]

        N = gt_depth.shape[0]
        # pred_depth[pred_depth != 0] = 1 / pred_depth[pred_depth != 0]

        if mask is None:
            mask = gt_depth > 0

        # Scale and shift prediction on disp before error calculation
        if len(pred_depth.shape) == 4:
            pred, gt_depth, mask = pred_depth.squeeze(1), gt_depth.squeeze(1), mask.squeeze(1)
        # depth -> disp
        gt_disp = gt_depth.clone()  # allocate new memory
        gt_disp[gt_disp != 0] = 1 / gt_disp[gt_disp != 0]
        scale, shift = compute_scale_and_shift(pred, gt_disp, mask) # scale shift on disp domain
        pred = (scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)) # not call by reference, new tensor

        # disp -> depth
        pred[pred != 0] = 1 / pred[pred != 0]

        pred = torch.clip(pred, 0, self.max_depth)

        mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_depth_metrics(gt_depth, pred, mask, max_depth=self.max_depth)

        self.metrics["err/mre"].update(mre, N)
        self.metrics["err/mae"].update(mae, N)
        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)


    def compute_eval_metrics(self, gt_depth, pred_depth, mask=None):
        """
        Computes metrics used to evaluate the model
        """
        # crop here
        if self.crop > 0:
            pred_depth = pred_depth[..., self.crop:-self.crop, :]
            gt_depth = gt_depth[..., self.crop:-self.crop, :]
            mask = mask[..., self.crop:-self.crop, :]
        N = gt_depth.shape[0]

        mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align)

        self.metrics["err/mre"].update(mre, N)
        self.metrics["err/mae"].update(mae, N)
        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)
