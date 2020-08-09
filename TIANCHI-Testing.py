#!/usr/bin/env python
# coding: utf-8

from model.UVOS import UVOS
from utils.DataLoading import DataDiscovering, DataReading

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

from itertools import combinations
import os
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda")

def test(root, pretrained_weight, Batch_Size=2):
    Dir = DataDiscovering(root)
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = Dir.reading_testing_data(random_subset=6)
    testset = DataReading(test_data,transform = transform)
    network = UVOS().cuda()
    network.load_state_dict(torch.load(pretrained_weight))
    
    test_length = len(list(test_data.keys()))
    for item in range(test_length):
        print("----- Reading testing folder data at {} -----".format(list(test_data.keys())[item]))
        Length = len(test_data[list(test_data.keys())[item]])
        Test_Data = list(combinations(testset[item],2))[:Length-1]
        length = len(Test_Data)
        if Length%Batch_Size != 0:
            Length = int(np.floor(length/Batch_Size)*Batch_Size)
            test_loader = DataLoader(Test_Data[:Length],batch_size=Batch_Size,shuffle=False,num_workers=4)
        else:
            test_loader = DataLoader(Test_Data,batch_size=Batch_Size,shuffle=False,num_workers=4)
        print("----- the total batches = {} -----".format(len(test_loader)))
        print("----- Fininsh Reading test data -----")

        for i,batch in enumerate(test_loader):
            Fa = batch[0]
            Fb = batch[1]
            pred_a, pred_b = network(Fa.to(device),Fb.to(device))
            pred_a = torch.transpose(pred_a[1],0,1)
            pred_a = torch.transpose(pred_a,1,2)
            plt.imshow(pred_a.cpu().detach())
            plt.savefig(root+"/Annotations/"+str(list(test_data.keys())[item])+"/{}.png".format(i))
            pred_b = torch.transpose(pred_b[1],0,1)
            pred_b = torch.transpose(pred_b,1,2)
            plt.imshow(pred_b.cpu().detach())
            plt.savefig(root+"/Annotations/"+str(list(test_data.keys())[item])+"/{}.png".format(i+1))


test(root = 'E:/TIANCHI', pretrained_weight = ".\saved_models\\val_saved_model_at_epoch_2_loss_0.026.pth",Batch_Size=2)





