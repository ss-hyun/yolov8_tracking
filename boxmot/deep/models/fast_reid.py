import sys
sys.path.insert(0, '/home/nextlab/sshyun/fast-reid')
import subprocess

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling.meta_arch.baseline import Baseline

CONFIG_FILE_PATH='/home/nextlab/sshyun/fast-reid/results/rotate-car/front/veri-yml/batch-x2/config.yaml'
MODEL_WEIGHTS='/home/nextlab/sshyun/fast-reid/results/rotate-car/front/veri-yml/batch-x2/model_best.pth'
MODEL_DEVICE='cuda:0'
    
        
def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE_PATH)
    cfg.merge_from_list(['MODEL.WEIGHTS', MODEL_WEIGHTS, 'MODEL.DEVICE', MODEL_DEVICE])
    cfg.freeze()
    return cfg


def main():
    cfg = setup()

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False 
    model = DefaultTrainer.build_model(cfg)

    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        
    return model


# class FastReID_R50_IBN(Baseline):
    
#     def __init__(
#         self,
#         num_classes=1000,
#         fc_dims=None
#     ):
#         self.backbone = main()
        
    


def fast_reid_R50_IBN(num_classes, loss='softmax', pretrained=True, **kwargs):
    # import logging
    # logger=logging.getLogger('ultralytics')
    model = main()
    # print(model)
    # raise Exception
    return model

# # encoding: utf-8
# """
# @author:  liaoxingyu
# @contact: sherlockliao01@gmail.com
# """
# import logging
# import math

# import torch
# from torch import nn
# import torch.nn.functional as F

# import torch.distributed as dist

# from collections import defaultdict
# from typing import Dict, List
# from termcolor import colored


# logger = logging.getLogger(__name__)

# def get_rank() -> int:
#     if not dist.is_available():
#         return 0
#     if not dist.is_initialized():
#         return 0
#     return dist.get_rank()
# def is_main_process() -> bool:
#     return get_rank() == 0


# def synchronize():
#     """
#     Helper function to synchronize (barrier) among all processes when
#     using distributed training
#     """
#     if not dist.is_available():
#         return
#     if not dist.is_initialized():
#         return
#     world_size = dist.get_world_size()
#     if world_size == 1:
#         return
#     dist.barrier()


# class BatchNorm(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
#                  bias_init=0.0, **kwargs):
#         super().__init__(num_features, eps=eps, momentum=momentum)
#         if weight_init is not None: nn.init.constant_(self.weight, weight_init)
#         if bias_init is not None: nn.init.constant_(self.bias, bias_init)
#         self.weight.requires_grad_(not weight_freeze)
#         self.bias.requires_grad_(not bias_freeze)


# class SyncBatchNorm(nn.SyncBatchNorm):
#     def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
#                  bias_init=0.0):
#         super().__init__(num_features, eps=eps, momentum=momentum)
#         if weight_init is not None: nn.init.constant_(self.weight, weight_init)
#         if bias_init is not None: nn.init.constant_(self.bias, bias_init)
#         self.weight.requires_grad_(not weight_freeze)
#         self.bias.requires_grad_(not bias_freeze)


# class IBN(nn.Module):
#     def __init__(self, planes, bn_norm, **kwargs):
#         super(IBN, self).__init__()
#         half1 = int(planes / 2)
#         self.half = half1
#         half2 = planes - half1
#         self.IN = nn.InstanceNorm2d(half1, affine=True)
#         self.BN = get_norm(bn_norm, half2, **kwargs)

#     def forward(self, x):
#         split = torch.split(x, self.half, 1)
#         out1 = self.IN(split[0].contiguous())
#         out2 = self.BN(split[1].contiguous())
#         out = torch.cat((out1, out2), 1)
#         return out


# class GhostBatchNorm(BatchNorm):
#     def __init__(self, num_features, num_splits=1, **kwargs):
#         super().__init__(num_features, **kwargs)
#         self.num_splits = num_splits
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))

#     def forward(self, input):
#         N, C, H, W = input.shape
#         if self.training or not self.track_running_stats:
#             self.running_mean = self.running_mean.repeat(self.num_splits)
#             self.running_var = self.running_var.repeat(self.num_splits)
#             outputs = F.batch_norm(
#                 input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
#                 self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
#                 True, self.momentum, self.eps).view(N, C, H, W)
#             self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
#             self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
#             return outputs
#         else:
#             return F.batch_norm(
#                 input, self.running_mean, self.running_var,
#                 self.weight, self.bias, False, self.momentum, self.eps)


# class FrozenBatchNorm(nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.
#     It contains non-trainable buffers called
#     "weight" and "bias", "running_mean", "running_var",
#     initialized to perform identity transformation.
#     The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
#     which are computed from the original four parameters of BN.
#     The affine transform `x * weight + bias` will perform the equivalent
#     computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
#     When loading a backbone model from Caffe2, "running_mean" and "running_var"
#     will be left unchanged as identity transformation.
#     Other pre-trained backbone models may contain all 4 parameters.
#     The forward is implemented by `F.batch_norm(..., training=False)`.
#     """

#     _version = 3

#     def __init__(self, num_features, eps=1e-5, **kwargs):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.register_buffer("weight", torch.ones(num_features))
#         self.register_buffer("bias", torch.zeros(num_features))
#         self.register_buffer("running_mean", torch.zeros(num_features))
#         self.register_buffer("running_var", torch.ones(num_features) - eps)

#     def forward(self, x):
#         if x.requires_grad:
#             # When gradients are needed, F.batch_norm will use extra memory
#             # because its backward op computes gradients for weight/bias as well.
#             scale = self.weight * (self.running_var + self.eps).rsqrt()
#             bias = self.bias - self.running_mean * scale
#             scale = scale.reshape(1, -1, 1, 1)
#             bias = bias.reshape(1, -1, 1, 1)
#             return x * scale + bias
#         else:
#             # When gradients are not needed, F.batch_norm is a single fused op
#             # and provide more optimization opportunities.
#             return F.batch_norm(
#                 x,
#                 self.running_mean,
#                 self.running_var,
#                 self.weight,
#                 self.bias,
#                 training=False,
#                 eps=self.eps,
#             )

#     def _load_from_state_dict(
#             self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#     ):
#         version = local_metadata.get("version", None)

#         if version is None or version < 2:
#             # No running_mean/var in early versions
#             # This will silent the warnings
#             if prefix + "running_mean" not in state_dict:
#                 state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
#             if prefix + "running_var" not in state_dict:
#                 state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

#         if version is not None and version < 3:
#             logger = logging.getLogger(__name__)
#             logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
#             # In version < 3, running_var are used without +eps.
#             state_dict[prefix + "running_var"] -= self.eps

#         super()._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#         )

#     def __repr__(self):
#         return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

#     @classmethod
#     def convert_frozen_batchnorm(cls, module):
#         """
#         Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
#         Args:
#             module (torch.nn.Module):
#         Returns:
#             If module is BatchNorm/SyncBatchNorm, returns a new module.
#             Otherwise, in-place convert module and return it.
#         Similar to convert_sync_batchnorm in
#         https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
#         """
#         bn_module = nn.modules.batchnorm
#         bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
#         res = module
#         if isinstance(module, bn_module):
#             res = cls(module.num_features)
#             if module.affine:
#                 res.weight.data = module.weight.data.clone().detach()
#                 res.bias.data = module.bias.data.clone().detach()
#             res.running_mean.data = module.running_mean.data
#             res.running_var.data = module.running_var.data
#             res.eps = module.eps
#         else:
#             for name, child in module.named_children():
#                 new_child = cls.convert_frozen_batchnorm(child)
#                 if new_child is not child:
#                     res.add_module(name, new_child)
#         return res


# def get_norm(norm, out_channels, **kwargs):
#     """
#     Args:
#         norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
#             or a callable that takes a channel number and returns
#             the normalization layer as a nn.Module
#         out_channels: number of channels for normalization layer

#     Returns:
#         nn.Module or None: the normalization layer
#     """
#     if isinstance(norm, str):
#         if len(norm) == 0:
#             return None
#         norm = {
#             "BN": BatchNorm,
#             "syncBN": SyncBatchNorm,
#             "GhostBN": GhostBatchNorm,
#             "FrozenBN": FrozenBatchNorm,
#             "GN": lambda channels, **args: nn.GroupNorm(32, channels),
#         }[norm]
#     return norm(out_channels, **kwargs)


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, int(channel / reduction), bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(channel / reduction), channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class Non_local(nn.Module):
#     def __init__(self, in_channels, bn_norm, reduc_ratio=2):
#         super(Non_local, self).__init__()

#         self.in_channels = in_channels
#         self.inter_channels = reduc_ratio // reduc_ratio

#         self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                            kernel_size=1, stride=1, padding=0)

#         self.W = nn.Sequential(
#             nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
#                       kernel_size=1, stride=1, padding=0),
#             get_norm(bn_norm, self.in_channels),
#         )
#         nn.init.constant_(self.W[1].weight, 0.0)
#         nn.init.constant_(self.W[1].bias, 0.0)

#         self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                                kernel_size=1, stride=1, padding=0)

#         self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                              kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         """
#                 :param x: (b, t, h, w)
#                 :return x: (b, t, h, w)
#         """
#         batch_size = x.size(0)
#         g_x = self.g(x).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)

#         theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
#         f = torch.matmul(theta_x, phi_x)
#         N = f.size(-1)
#         f_div_C = f / N

#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *x.size()[2:])
#         W_y = self.W(y)
#         z = W_y + x
#         return z

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
#                  stride=1, downsample=None, reduction=16):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         if with_ibn:
#             self.bn1 = IBN(planes, bn_norm)
#         else:
#             self.bn1 = get_norm(bn_norm, planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = get_norm(bn_norm, planes)
#         self.relu = nn.ReLU(inplace=True)
#         if with_se:
#             self.se = SELayer(planes, reduction)
#         else:
#             self.se = nn.Identity()
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.se(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
#                  stride=1, downsample=None, reduction=16):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         if with_ibn:
#             self.bn1 = IBN(planes, bn_norm)
#         else:
#             self.bn1 = get_norm(bn_norm, planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = get_norm(bn_norm, planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = get_norm(bn_norm, planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         if with_se:
#             self.se = SELayer(planes * self.expansion, reduction)
#         else:
#             self.se = nn.Identity()
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.se(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):
#     def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
#         self.inplanes = 64
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = get_norm(bn_norm, 64)
#         self.relu = nn.ReLU(inplace=True)
#         # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#         self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
#         self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
#         self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
#         self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)

#         self.random_init()

#         # fmt: off
#         if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
#         else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
#         # fmt: on

#     def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 get_norm(bn_norm, planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

#         return nn.Sequential(*layers)

#     def _build_nonlocal(self, layers, non_layers, bn_norm):
#         self.NL_1 = nn.ModuleList(
#             [Non_local(256, bn_norm) for _ in range(non_layers[0])])
#         self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
#         self.NL_2 = nn.ModuleList(
#             [Non_local(512, bn_norm) for _ in range(non_layers[1])])
#         self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
#         self.NL_3 = nn.ModuleList(
#             [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
#         self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
#         self.NL_4 = nn.ModuleList(
#             [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
#         self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         # layer 1
#         NL1_counter = 0
#         if len(self.NL_1_idx) == 0:
#             self.NL_1_idx = [-1]
#         for i in range(len(self.layer1)):
#             x = self.layer1[i](x)
#             if i == self.NL_1_idx[NL1_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_1[NL1_counter](x)
#                 NL1_counter += 1
#         # layer 2
#         NL2_counter = 0
#         if len(self.NL_2_idx) == 0:
#             self.NL_2_idx = [-1]
#         for i in range(len(self.layer2)):
#             x = self.layer2[i](x)
#             if i == self.NL_2_idx[NL2_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_2[NL2_counter](x)
#                 NL2_counter += 1

#         # layer 3
#         NL3_counter = 0
#         if len(self.NL_3_idx) == 0:
#             self.NL_3_idx = [-1]
#         for i in range(len(self.layer3)):
#             x = self.layer3[i](x)
#             if i == self.NL_3_idx[NL3_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_3[NL3_counter](x)
#                 NL3_counter += 1

#         # layer 4
#         NL4_counter = 0
#         if len(self.NL_4_idx) == 0:
#             self.NL_4_idx = [-1]
#         for i in range(len(self.layer4)):
#             x = self.layer4[i](x)
#             if i == self.NL_4_idx[NL4_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_4[NL4_counter](x)
#                 NL4_counter += 1

#         return x

#     def random_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


# def _group_to_str(group: List[str]) -> str:
#     """
#     Format a group of parameter name suffixes into a loggable string.

#     Args:
#         group (list[str]): list of parameter name suffixes.
#     Returns:
#         str: formated string.
#     """
#     if len(group) == 0:
#         return ""

#     if len(group) == 1:
#         return "." + group[0]

#     return ".{" + ", ".join(group) + "}"


# def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
#     """
#     Group keys based on common prefixes. A prefix is the string up to the final
#     "." in each key.

#     Args:
#         keys (list[str]): list of parameter names, i.e. keys in the model
#             checkpoint dict.
#     Returns:
#         dict[list]: keys with common prefixes are grouped into lists.
#     """
#     groups = defaultdict(list)
#     for key in keys:
#         pos = key.rfind(".")
#         if pos >= 0:
#             head, tail = key[:pos], [key[pos + 1:]]
#         else:
#             head, tail = key, []
#         groups[head].extend(tail)
#     return groups


# def get_missing_parameters_message(keys: List[str]) -> str:
#     """
#     Get a logging-friendly message to report parameter names (keys) that are in
#     the model but not found in a checkpoint.

#     Args:
#         keys (list[str]): List of keys that were not found in the checkpoint.
#     Returns:
#         str: message.
#     """
#     groups = _group_checkpoint_keys(keys)
#     msg = "Some model parameters or buffers are not found in the checkpoint:\n"
#     msg += "\n".join(
#         "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
#     )
#     return msg


# def get_unexpected_parameters_message(keys: List[str]) -> str:
#     """
#     Get a logging-friendly message to report parameter names (keys) that are in
#     the checkpoint but not found in the model.

#     Args:
#         keys (list[str]): List of keys that were not found in the model.
#     Returns:
#         str: message.
#     """
#     groups = _group_checkpoint_keys(keys)
#     msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
#     msg += "\n".join(
#         "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
#     )
#     return msg


# def fast_reid_R50_IBN(num_classes, loss='softmax', pretrained=True, **kwargs):
#     with_ibn = True
#     with_se=False
#     with_nl=True
    
#     model = ResNet(
#         last_stride=1, 
#         bn_norm="BN", 
#         with_ibn=with_ibn,
#         with_se=with_se,
#         with_nl=with_nl, 
#         block=Bottleneck,
#         layers=[3, 4, 6, 3], 
#         non_layers=[0, 2, 3, 0]
#     )
#     if pretrained:
#         # Load pretrain path if specifically
#         if pretrain_path:
#             try:
#                 state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
#                 logger.info(f"Loading pretrained model from {pretrain_path}")
#             except FileNotFoundError as e:
#                 logger.info(f'{pretrain_path} is not found! Please check this path.')
#                 raise e
#             except KeyError as e:
#                 logger.info("State dict keys error! Please check the state dict.")
#                 raise e
#         else:
#             raise ValueError("Please specify the pretrain path!")

#         incompatible = model.load_state_dict(state_dict, strict=False)
#         if incompatible.missing_keys:
#             logger.info(
#                 get_missing_parameters_message(incompatible.missing_keys)
#             )
#         if incompatible.unexpected_keys:
#             logger.info(
#                 get_unexpected_parameters_message(incompatible.unexpected_keys)
#             )

#     return model
