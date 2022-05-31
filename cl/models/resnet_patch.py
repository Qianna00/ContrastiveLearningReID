from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer, GeneralizedMeanPoolingP


__all__ = ['ResNetP', 'resnetp18', 'resnetp34', 'resnetp50', 'resnetp101',
           'resnetp152']


class ResNetP(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=True,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(ResNetP, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNetP.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetP.__factory[depth]()
        if pretrained:
            state_dict = torch.load('/root/data/zq/pretrained_models/resnet50-0676ba61.pth')
            resnet.load_state_dict(state_dict)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        # self.gap = build_pooling_layer(pooling_type)
        self.gap = GeneralizedMeanPoolingP(norm=3, output_size=(4, 9), eps=1e-6, extra=True)
        self.gap_p = GeneralizedMeanPoolingP(norm=3, output_size=1, eps=1e-6, extra=False)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        bs = x.size(0)
        x = self.base(x)

        # print(type(x), len(x) if isinstance(x, list) else x.size())

        x, x_extra = self.gap(x)  # x: 32*2048*1*1  x_extra: 32*2048*4*9  imgs_per_gpu: 32
        x = x.view(x.size(0), -1)  # x: 32*2048
        x_b = x_extra[:, :, -1:, :]  # x_b: 32*2048*1*9
        x_tl = x_extra[:, :, :3, :3]  # x_tn: 32*2048*3*3
        x_tm = x_extra[:, :, :3, 3:6]
        x_tr = x_extra[:, :, :3, 3:6]

        x_b = self.gap_p(x_b).view(x_b.size(0), -1)
        x_tl = self.gap_p(x_tl).view(x_b.size(0), -1)
        x_tm = self.gap_p(x_tm).view(x_b.size(0), -1)
        x_tr = self.gap_p(x_tr).view(x_b.size(0), -1)

        # print(x_b.size(), x_tl.size(), x_tm.size(), x_tr.size())

        if self.cut_at_pooling:
            # return x
            return x, x_b, x_tl, x_tm, x_tr

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnetp18(**kwargs):
    return ResNetP(18, **kwargs)


def resnetp34(**kwargs):
    return ResNetP(34, **kwargs)


def resnetp50(**kwargs):
    return ResNetP(50, **kwargs)


def resnetp101(**kwargs):
    return ResNetP(101, **kwargs)


def resnetp152(**kwargs):
    return ResNetP(152, **kwargs)
