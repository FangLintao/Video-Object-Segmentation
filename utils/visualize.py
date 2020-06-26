#!/usr/bin/env python
# coding: utf-8

"""
Created on Mon Jun 22 14:48:14 2020

@author: Michael Fang
"""

import matplotlib.pyplot as plt
import torch
import cv2

class visualize:
    def __init__(self):
        pass
    def dataloading_visualize(self, dataloader):
        """
        Showing dataloading distribution, for vivid visulization. 
        
        Inputs:
            dataloader: train_loader or text_loader
            address_loader: train_address_loader or test_address_loader
        """
        batch_size = len(dataloader)
        if batch_size >=8:
            batch_size = 8
        fig = plt.figure(figsize=(32, 32))
        
        for i in range(batch_size):
            ax1 = plt.subplot(batch_size,2,2*i+1)
            ax2 = plt.subplot(batch_size,2,2*i+2)
            
            Img_one = torch.transpose(torch.transpose(dataloader[i][0],1,0),2,1)
            Img_one = cv2.cvtColor(Img_one.numpy(),cv2.COLOR_BGR2RGB)
            #ax1.set_title(add_img[0][i])
            ax1.imshow(Img_one)
            
            label_one = torch.transpose(torch.transpose(dataloader[i][1],1,0),2,1)
            label_one = cv2.cvtColor(label_one.numpy(),cv2.COLOR_BGR2RGB)
            #ax2.set_title(add_label[0][i])
            #label_one = dataset[i][1].squeeze()
            ax2.imshow(label_one)