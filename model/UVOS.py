#!/usr/bin/env python
# coding: utf-8

"""
Reference: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks

"""

import torch.nn as nn
import torch.nn.functional as F
import torch 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
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
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        out = self.relu(out)

        return out


class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv= nn.Conv2d(2048, depth, 1,1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(2048, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0], dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1], dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2], dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d( depth*5, 256, kernel_size=3, padding=1 ) 
        self.bn = nn.BatchNorm2d(256)
        self.prelu = nn.PReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size=x.shape[2:]
        image_features=self.mean(x)
        image_features=self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features=F.interpolate(image_features, size=size, mode='bilinear',align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0) 
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1) 
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2) 
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3) 
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        
        return out
    
    
class DeepLab(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(DeepLab, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(ASPP, [ 6, 12, 18], [6, 12, 18], 512)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, depth):
        return block(dilation_series, padding_series, depth)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.layer5(x)
        return output, input_size


class CoAttNet(nn.Module):
    def  __init__(self, all_channel=256):
        super(CoAttNet, self).__init__()
        self.all_channel = all_channel
        self.linear = nn.Linear(self.all_channel, self.all_channel,bias = False)
        self.gate = nn.Conv2d(self.all_channel, 1, kernel_size  = 1, bias = False)
        self.gate_sig = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)  

    def forward(self, Va_output, Vb_output, input_size): 

        fea_size = Vb_output.size()[2:]
        all_dim = fea_size[0]*fea_size[1]
        # Vanilla co-attentaion
        Va_output_flat = Va_output.view(-1, Vb_output.size()[1], all_dim) # N,C,H*W
        Vb_output_flat = Vb_output.view(-1, Vb_output.size()[1], all_dim) # N,C,H*W
        Va_output_t = torch.transpose(Va_output_flat,1,2).contiguous()  # N,H*W,C
        Va_output_corr = self.linear(Va_output_t) # Vb.T * W
        S = torch.bmm(Va_output_corr, Vb_output_flat) # S = Vb.T * W * Va
        S_c = F.softmax(S.clone(), dim = 1) #
        S_r = F.softmax(torch.transpose(S,1,2),dim=1)
        Vb_output_att = torch.bmm(Va_output_flat, S_r).contiguous()
        Va_output_att = torch.bmm(Vb_output_flat, S_c).contiguous()
        # Gated co-attention
        Va_att = Va_output_att.view(-1, Vb_output.size()[1], fea_size[0], fea_size[1])  
        Vb_att = Vb_output_att.view(-1, Vb_output.size()[1], fea_size[0], fea_size[1])
        Va_mask = self.gate(Va_att)
        Vb_mask = self.gate(Vb_att)
        Va_mask = self.gate_sig(Va_mask)
        Vb_mask = self.gate_sig(Vb_mask)
        Va_att = Va_att * Va_mask
        Vb_att = Vb_att * Vb_mask
        Va_att = torch.cat([Va_att, Va_output],1) 
        Vb_att = torch.cat([Vb_att, Vb_output],1)

        return Va_att,Vb_att,input_size

class SegNet(nn.Module):
    def  __init__(self, num_classes = 3, all_channel=256, all_dim=60*60):
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        self.all_channel = all_channel
        self.conv1 = nn.Conv2d(self.all_channel*2, self.all_channel, kernel_size=3, padding=1, bias = False)
        self.conv2 = nn.Conv2d(self.all_channel*2, self.all_channel, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.all_channel)
        self.bn2 = nn.BatchNorm2d(self.all_channel)
        self.relu = nn.ReLU(inplace=True)
        self.classifier1 = nn.Conv2d(self.all_channel, self.num_classes, kernel_size=1, bias = True)
        self.classifier2 = nn.Conv2d(self.all_channel, self.num_classes, kernel_size=1, bias = True)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    

    def forward(self, Va_att, Vb_att,input_size): 
        Va_att  = self.conv1( Va_att )
        Vb_att  = self.conv2( Vb_att ) 
        Va_att  = self.bn1( Va_att )
        Vb_att  = self.bn2( Vb_att )
        Va_att  = self.relu( Va_att )
        Vb_att  = self.relu( Vb_att )
        Ya = self.classifier1(Va_att)
        Yb = self.classifier2(Vb_att)   
        Ya = F.interpolate(Ya, input_size, mode='bilinear',align_corners=True)
        Yb = F.interpolate(Yb, input_size, mode='bilinear',align_corners=True) 
        Ya = self.sigmoid(Ya)
        Yb = self.sigmoid(Yb)
        return Ya, Yb


class UVOS(nn.Module):
    def __init__(self,block = Bottleneck,layers = [3, 4, 6, 3]):
        super(UVOS,self).__init__()
        self.encoder = DeepLab(block, layers)
        self.co_att = CoAttNet()
        self.seg = SegNet()
    def forward(self, Fa,Fb):
        Va,_ = self.encoder(Fa)
        Vb,input_size = self.encoder(Fb)
        Za,Zb,input_size = self.co_att(Va,Vb,input_size)
        Oa, Ob = self.seg(Za,Zb,input_size)
        return Oa,Ob

