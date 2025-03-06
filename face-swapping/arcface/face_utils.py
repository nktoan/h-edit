import torch
import torch.nn as nn
from torch.nn import functional as F

def encode_segmentation(segmentation, no_neck=True):
    # parse = segmentation[:,:,0]

    face_part_ids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 10
    hair_id = 13
    face_map = torch.zeros_like(segmentation)
    mouth_map = torch.zeros_like(segmentation)
    hair_map = torch.zeros_like(segmentation)

    for valid_id in face_part_ids:
        valid_index = torch.where(segmentation==valid_id)
        face_map[valid_index] = 1
    valid_index = torch.where(segmentation==mouth_id)
    mouth_map[valid_index] = 1
    valid_index = torch.where(segmentation==hair_id)
    hair_map[valid_index] = 1

    out = torch.cat([face_map, mouth_map, hair_map], axis=1)
    return out

class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask