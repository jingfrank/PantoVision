"""
HRNet implementation for Pantograph Keypoint Localization.
Based on "Deep High-Resolution Representation Learning for Visual Recognition" (CVPR 2019 / TPAMI 2020)
Paper: https://arxiv.org/abs/1908.07919
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, momentum=BN_MOMENTUM
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            raise ValueError(f"num_branches={num_branches} != len(num_blocks)")
        if num_branches != len(num_channels):
            raise ValueError(f"num_branches={num_branches} != len(num_channels)")
        if num_branches != len(num_inchannels):
            raise ValueError(f"num_branches={num_branches} != len(num_inchannels)")

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNetKeypoint(nn.Module):
    """
    HRNet Keypoint Localization Model.
    Maintains high resolution representation throughout the entire network.
    """

    def __init__(self, num_joints=10, width=32, in_channels=3):
        super(HRNetKeypoint, self).__init__()
        self.inplanes = 64
        self.num_joints = num_joints
        self.width = width

        # Stem network
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        # Stage 2
        w = width
        stage2_channels = [w, w * 2]
        self.transition1 = self._make_transition_layer([256], stage2_channels)
        self.stage2, pre_channels = self._make_stage(
            num_modules=1,
            num_branches=2,
            num_blocks=[4, 4],
            num_channels=stage2_channels,
            block=BasicBlock,
            in_channels=stage2_channels,
        )

        # Stage 3
        stage3_channels = [w, w * 2, w * 4]
        self.transition2 = self._make_transition_layer(pre_channels, stage3_channels)
        self.stage3, pre_channels = self._make_stage(
            num_modules=4,
            num_branches=3,
            num_blocks=[4, 4, 4],
            num_channels=stage3_channels,
            block=BasicBlock,
            in_channels=stage3_channels,
        )

        # Stage 4
        stage4_channels = [w, w * 2, w * 4, w * 8]
        self.transition3 = self._make_transition_layer(pre_channels, stage4_channels)
        self.stage4, pre_channels = self._make_stage(
            num_modules=3,
            num_branches=4,
            num_blocks=[4, 4, 4, 4],
            num_channels=stage4_channels,
            block=BasicBlock,
            in_channels=stage4_channels,
            multi_scale_output=False,
        )

        # Keypoint Head (Heatmap predictor on highest resolution branch)
        self.final_layer = nn.Conv2d(
            in_channels=pre_channels[0],
            out_channels=num_joints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_inchannels, num_outchannels):
        num_branches_pre = len(num_inchannels)
        num_branches_cur = len(num_outchannels)
        transition_layers = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_inchannels[i] != num_outchannels[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[i],
                                num_outchannels[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_outchannels[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inch = num_inchannels[-1]
                    outch = num_outchannels[i] if j == i - num_branches_pre else inch
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inch, outch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outch, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        num_modules,
        num_branches,
        num_blocks,
        num_channels,
        block,
        in_channels,
        multi_scale_output=True,
    ):
        modules = []
        for i in range(num_modules):
            reset_multiscale = (
                multi_scale_output if i == num_modules - 1 else True
            )
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    fuse_method="SUM",
                    multi_scale_output=reset_multiscale,
                )
            )
            in_channels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), in_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # Stage 2
        xl = []
        for i in range(2):
            if self.transition1[i] is not None:
                xl.append(self.transition1[i](x))
            else:
                xl.append(x)
        y_list = self.stage2(xl)

        # Stage 3
        xl = []
        for i in range(3):
            if self.transition2[i] is not None:
                xl.append(self.transition2[i](y_list[-1]))
            else:
                xl.append(y_list[i])
        y_list = self.stage3(xl)

        # Stage 4
        xl = []
        for i in range(4):
            if self.transition3[i] is not None:
                xl.append(self.transition3[i](y_list[-1]))
            else:
                xl.append(y_list[i])
        y_list = self.stage4(xl)

        # Final keypoint prediction head on highest resolution branch (1/4 size)
        out = self.final_layer(y_list[0])
        return out


def decode_heatmaps(heatmaps):
    """
    Decodes keypoint locations from heatmaps with subpixel accuracy.
    heatmaps: torch.Tensor of shape (B, K, H, W) or numpy array
    Returns: coords (B, K, 2), maxvals (B, K, 1) in heatmap coordinate space
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()

    batch_size, num_joints, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % w
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / w)

    # Subpixel refinement using dark pose / Taylor expansion approximation
    for b in range(batch_size):
        for j in range(num_joints):
            hm = heatmaps[b, j]
            px = int(preds[b, j, 0])
            py = int(preds[b, j, 1])
            if 1 < px < w - 1 and 1 < py < h - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px],
                    ]
                )
                preds[b, j] += np.sign(diff) * 0.25

    return preds, maxvals
