import torch
import roipool3d_cuda
import numpy as np


def roipool3d_gpu(pts, boxes3d, pool_extra_width, sampled_pt_num=512):
    """
    :param pts: (B, N, 3)
    :param boxes3d: (B, M, 7)
    :param pool_extra_width: float
    :param sampled_pt_num: int
    :return:
        pooled_features: (B, M, 512, 3)
        pooled_empty_flag: (B, M)
    """
    batch_size, boxes_num = pts.shape[0], boxes3d.shape[1]
    pooled_boxes3d = enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)

    pooled_features = torch.cuda.FloatTensor(torch.Size((batch_size, boxes_num, sampled_pt_num, 3))).zero_()
    pooled_empty_flag = torch.cuda.IntTensor(torch.Size((batch_size, boxes_num))).zero_()

    roipool3d_cuda.forward(pts.contiguous(), pooled_boxes3d.contiguous(), pooled_features, pooled_empty_flag)

    return pooled_features, pooled_empty_flag

def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d