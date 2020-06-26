#!/usr/bin/env python
# coding: utf-8

"""
Created on Mon Jun 22 14:48:14 2020

@author: Michael Fang
"""

import os
import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

class DataDiscovering:
    """
    By discovering TIANCHI dataset, it includes 850 small videos with different numbers of frames for each samll video.
    However, the total number frames in each small video only has around 150 frames. Training data and testing data based on train.txt & test.txt in ImageSets.
    TIANCHI data files includes Annotation, ImageSets and JPEGImages.
    
    Anotation file: object masks. Only one frame for testing object mask!!
    ImageSets: 800 training folder and 50 testing folder
    JPEGImages: 850 small videos with frames
    
    Inputs:
        root(str): the address of TIANCHI dataset
        
    """
    def __init__(self,root):
        self.root = root
        self.annotation={}
        
    def folders(self):
        """offering the list of files' names
        """
        self.path = os.path.join(self.root +'/JPEGImages')
        self.files = os.listdir(self.path)
        return self.files
    
    def load_annotation(self):
        """
        loading annotation dataset
        Outputs:
            self.annotation(dict): keys -> filenames, values -> images' address
        """
        path = os.path.join(self.root +'/Annotations')
        files = os.listdir(path)
        for subFile in tqdm(files,ascii=True,desc="loading annotation data"):
            file_path = os.path.join(path+'/'+subFile)
            image_items = os.listdir(file_path)
            image_dirs = [os.path.join(file_path + '/' + image) for image in image_items]
            self.annotation[subFile] = image_dirs
            
    def reading_data(self,train_val_split=0.8,random_subset=None):
        """
        By reference to train.txt and test.txt in /ImageSets, train.txt offers folders' name poining to training images and so does text.txt. 
        Because different folders contain different numbers of images, and each folder has differnt contents showing in their video frames, In this case, training data and text data are in dictory type in order to load all of image information.
        Here is one thing on annotation that for training images, annotation folder offers all of mask for training images,while for testing image, it only offers the first frame mask to testing image.
        Inputs:
            train_val_split(scale): spliting into training and valid dataset
            random_subset(scale): randomly choose subset folders in training datatset. Because the total video is quit large and after combination, the overall image groups are more than millions, then its training is super expensive.
        Outputs:
            train_data(dict):  only training image address with its corresponding annotation mask image address
            val_data(dict):  only valid image address with its corresponding annotation mask image address
            test_data(dict): the first one is annotation mask image and for the rest,  tehy are testing image address
        """
        train_data = {}
        val_data = {}
        test_data = {}
        self.load_annotation()
        path = os.path.join(self.root +'/ImageSets')
        files = os.listdir(path)
        for subFile in tqdm(files,ascii=True,desc="loading train&test data"):
            if('train.txt' in subFile):
                file_path = open(path+'/'+subFile)
                iter_file = list(iter(file_path))
                if random_subset is not None:
                    iter_file = np.random.choice(iter_file,100)
                Len = len(iter_file)
                train_num = int(Len * train_val_split)
                train_iter_file = iter_file[:train_num]
                val_iter_file = iter_file[train_num:]
                for line in train_iter_file:
                    line = line[0:6]
                    train_temp = []
                    train_images_path = os.path.join(self.root + '/JPEGImages/'+line)
                    train_image_items = os.listdir(train_images_path)
                    for idx,train_image in enumerate(train_image_items):
                        train_image_item = os.path.join(train_images_path + '/' + train_image)
                        train_temp.append([train_image_item,self.annotation[line][idx]])
                    train_data[line] = train_temp
                for line in val_iter_file:
                    line = line[0:6]
                    val_temp = []
                    val_images_path = os.path.join(self.root + '/JPEGImages/'+line)
                    val_image_items = os.listdir(val_images_path)
                    for idx,val_image in enumerate(val_image_items):
                        val_image_item = os.path.join(val_images_path + '/' + val_image)
                        val_temp.append([val_image_item,self.annotation[line][idx]])
                    val_data[line] = val_temp
            elif('test.txt' in subFile):
                file_path = open(path+'/'+subFile)
                iter_file = iter(file_path)
                for line in iter_file:
                    line = line[0:6]
                    test_temp = []
                    test_temp.append(self.annotation[line][0])
                    test_images_path = os.path.join(self.root + '/JPEGImages/'+line)
                    test_image_items = os.listdir(test_images_path)
                    for idx,test_image in enumerate(test_image_items):
                        test_image_item = os.path.join(test_images_path + '/' + test_image)
                        test_temp.append(test_image_item)
                    test_data[line] = test_temp
            else:
                pass
        return train_data, val_data, test_data

class DataReading(Dataset):
    """
    Associating with torch.nn.data.Dataset, it allows us to use torch.nn.data.DataLoader for the following training stage. 
    By using training data and testing data address, images in each folder can be introduced to trainset and testset. 
    Each image will be resized into shape(320,180).
    
    Input:
        data_file: train data or test data
    """
    def __init__(self,data_file,transform=None,size=(320,180)):
        self.data_file = data_file
        self.transform = transform
        self.size = size
        self.items = list(self.data_file.keys())
        
    def __getitem__(self, idx):
        """
        According to paper" Unsupervised Video Object Segmentation with Co-Attention Siamese Networks", for the same video, one pair of frames is needed and randomly choosen from datatset. By using combination, all of image pairs are covered. 
        Outputs:
            image_item(list): all pairs of images and annotation masks
                        e.g.[ [ [tensor(train_image_one),tensor(train_image_two)],                                                          [tensor(annot_image_one),tensor(annot_image_two)] ],
                                ........
                            [ [tensor(train_image_N),tensor(train_image_N)],                                                              [tensor(annot_image_N),tensor(annot_image_N)] ]  ]
        """
        image_item = []
        sel_item = self.items[idx]
        for path in self.data_file[sel_item]:
            img_add = path[0]
            lab_add = path[1]
            image = cv2.imread(img_add)
            label = cv2.imread(lab_add)
            image = cv2.resize(image,self.size)
            label = cv2.resize(label,self.size)
            image = self.transform(image)
            label = self.transform(label)
            #label[0] = label[0]*50
            #label[1] = label[1]*120
            #label[2] = label[2]*240
            #label = (torch.sum(label,0,keepdim=True)).long()
            image_item.append([image,label])
        return image_item
    
    def length(self):
        """
        Giving basic information about total number of folders and total number of images for each folder.
        
        Outputs:
            length(dict)
        """
        length = {}
        length["total video"] = len(self.items)
        for item in self.items:
            length[item] = len(self.data_file[item])
        return length