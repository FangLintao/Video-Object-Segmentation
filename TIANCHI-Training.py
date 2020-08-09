#!/usr/bin/env python
# coding: utf-8

from model.UVOS import UVOS
from utils.DataLoading import DataDiscovering, DataReading
from utils.tensorboard_evaluation import Evaluation

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

from itertools import combinations
from tqdm import tqdm
import os
import argparse
import numpy as np

device = torch.device("cuda")

def train(root, num_epoch, Batch_Size, Random_Subset = None, learning_rate = 2.5e-4, model_dir="./saved_models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    
    #tensorboard_Loss = Evaluation(tensorboard_dir,name = "Loss",stats=["val_Loss", "train_Loss"])
    #tensorboard_Accuracy = Evaluation(tensorboard_dir,name = "Accuracy",stats=["val_accuracy"])
    
    Dir = DataDiscovering(root)
    transform = transforms.Compose([transforms.ToTensor()])
    train_data, val_data = Dir.reading_training_data(train_val_split=0.7,random_subset=Random_Subset)
    trainset = DataReading(train_data,transform = transform)
    valset = DataReading(val_data,transform = transform)

    #DL_network = DeepLab().cuda()
    network = UVOS().cuda()
    
    optimizer = optim.SGD( network.parameters(), lr=2.5e-4,momentum=0.9, weight_decay=1e-4 )
    loss_fn = nn.L1Loss()
    #for epoch in tqdm(range(num_epoch),ascii=True,desc="epoch -> "):
    for epoch in range(num_epoch):
        print("-----------------------------epoch {}-----------------------------".format(epoch))
        train_length = len(list(train_data.keys()))
        train_loss = 0
        for item in range(train_length):
            print("----- Reading training data -----")
            training_data = list(combinations(trainset[item],2))
            length = len(training_data)
            if length%Batch_Size != 0:
                Length = int(np.floor(length/Batch_Size)*Batch_Size)
                train_loader = DataLoader(training_data[:Length],batch_size=Batch_Size,shuffle=False,num_workers=4)
            else:
                train_loader = DataLoader(training_data,batch_size=Batch_Size,shuffle=False,num_workers=4)
            print("----- the total batches = {} -----".format(len(train_loader)))
            print("----- Fininsh Reading training data -----")
            folder_loss = 0
            count = 0
            #tensorboard_training_folder_Loss = Evaluation(tensorboard_dir,name = "training_the_folder_{}".format(list(train_data.keys())[item]),stats=["folder_train_Loss"])
            for batch in train_loader:
                Fa = batch[0][0]
                Fb = batch[1][0]
                Fa_label = batch[0][1].squeeze()
                Fb_label = batch[1][1].squeeze()
                optimizer.zero_grad()
                pred_a, pred_b = network(Fa.to(device),Fb.to(device))
                Loss = loss_fn(pred_a,Fa_label.to(device)) + loss_fn(pred_b, Fb_label.to(device))
                del Fa, Fb, Fa_label, Fb_label
                Loss.backward()
                optimizer.step()
                folder_loss += Loss
                #tensorboard_training_folder_Loss.write_episode_data(count,{"folder_train_Loss":Loss.cpu().detach()})
                count += 1
            train_loss += folder_loss/count
            del training_data, train_loader
            print("-----training stage folder loss = {}-----".format(folder_loss/count))
        train_loss = train_loss / train_length
        torch.save(network.state_dict(), os.path.join(model_dir+'/'+ "train_saved_model_at_epoch_{}_loss={}.pth".format(epoch,train_loss)))
        print("-----training stage folder loss = {}-----".format(epoch,train_loss))
        #tensorboard_Loss.write_episode_data(epoch,{"train_Loss":train_loss.cpu().detach()})
        
        with torch.no_grad():
            val_length = len(list(val_data.keys()))
            val_loss = 0
            val_accuracy = 0
            for item in range(val_length):
                print("----- Reading validation data -----")
                valid_data = list(combinations(valset[item],2))
                length = len(valid_data)
                if length%Batch_Size != 0:
                    Length = int(np.floor(length/Batch_Size)*Batch_Size)
                    val_loader = DataLoader(valid_data[:Length],batch_size=Batch_Size,shuffle=False,num_workers=4)
                else:
                    val_loader = DataLoader(valid_data,batch_size=Batch_Size,shuffle=False,num_workers=4)
                print("----- the total batches = {} -----".format(len(val_loader)))
                print("----- Fininsh Reading validation data -----")
                folder_loss = 0
                count = 0
                #tensorboard_val_folder_Loss = Evaluation(tensorboard_dir,name = "val_the_folder_{}".format(list(val_data.keys())[item]),stats=["folder_val_Loss"])
                for batch in val_loader:
                    Fa = batch[0][0]
                    Fb = batch[1][0]
                    Fa_label = batch[0][1].squeeze()
                    Fb_label = batch[1][1].squeeze()
                    pred_a, pred_b = network(Fa.to(device),Fb.to(device))
                    Loss = loss_fn(pred_a,Fa_label.to(device)) + loss_fn(pred_b, Fb_label.to(device))
                    del Fa, Fb, Fa_label, Fb_label
                    #tensorboard_val_folder_Loss.write_episode_data(count,{"folder_val_Loss":Loss.cpu().detach()})
                    folder_loss += Loss
                    count += 1
                #tensorboard_val_folder_Loss.save("val_the_folder_{}".format(list(val_data.keys())[item]))
                val_loss += folder_loss/count
                del valid_data, val_loader
                print("-----val stagit pullge folder loss = {}-----".format(folder_loss/count))
            val_loss = val_loss / val_length
            print("-----epoch {}-- val stage folder loss = {}-----".format(epoch,val_loss))
            #if val_loss <= 0.2 :
            torch.save(network.state_dict(), os.path.join(model_dir +'/'+"val_saved_model_at_epoch_{}_loss_{}.pth".format(epoch,val_loss)))
            print("-----Model saved in file-----")
            #tensorboard_Loss.write_episode_data(epoch,{"val_Loss":val_loss.cpu().detach()})


train(root = 'E:/TIANCHI', num_epoch = 10, Batch_Size=2,Random_Subset = 6)

