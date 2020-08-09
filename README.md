# Video-Object-Segmentation [currently working]
![image](https://github.com/FangLintao/Video-Object-Segmentation/blob/master/images/cover.png)
## 1. Introduction
Video Object Segmentation is to track objects in video and segment objects from video. Video Object Segmentation is one of main techs in computer vision and it has widely-used field in practical application.
## 2. Dataset

    Download Link: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531797/tianchiyusai.zip
    
Ⅰ. size: 19.06G  
Ⅱ. Contains: Annotation, ImageSets, JPEGImages  
Ⅲ. train & test datasets  
For brief demostration, running [DataLoading_demo.ipynb](https://github.com/FangLintao/Video-Object-Segmentation/blob/master/utils/DataLoading_demo.ipynb)
### 2.1 Data Discovering
total number of videos is 850, 800 for training videos and 50 for testing video. The specific folders are shown at train.txt and text.txt in ImageSets, which can load video frames for training dataset and test dataset. Additionally, for testing stage, testing annotation folders only the first frame.  In data discovery, training dataset is splited into training and valid datasets for training stage.  In order to get train&val&test dataset, running the following code

    from utils.DataLoading import DataDiscovering
    train_data, val_data, test_data = Dir.reading_data()

Ⅰ. For training&val datasets => contains addresses of frames coupled with corresponding annotation frames as one group;   
Ⅱ. For testing dataset => containing addresses of  the first annotation frame and following testing frames  
### 2.2 Data Reading
Data reading is capable of reading images based on their addresses. By running below codes  

    from utils.DataLoading import Datareading
    train_set = DataReading(train_data,transform = transform)
    val_set = DataReading(val_data,transform = transform)

e.g. train_set[i] => the i th folder   
To visualize the images distribution in train_set and val_set, running below codes

    from visualize import visualize
    Vis = visualize()
    Vis.dataloading_visualize(train_set[i])

## 3. [Architecture](https://github.com/FangLintao/Video-Object-Segmentation/blob/master/model/UVOS.py)
![image](https://github.com/FangLintao/Video-Object-Segmentation/blob/master/images/1.png)  
###### reference: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks,Xiankai Lu,Wenguan Wang,Chao Ma,Jianbing Shen,Ling Shao,Fatih Porikli
##### Main Characteristics
Ⅰ.DeepLabV3 offers static analysis on video frames while co-attention offers dynamic analysis on pairs of video frame;s  
Ⅱ. Co-attention = channel-wise attention + gate attention;  
Ⅲ. Using co-attention algorithm to fuse feature maps infromation in pairs of frames from the same video so that objects in one video can be tracked and locked, which is convenient to segment objects from background;  
### 3.1 Architecture Components
Ⅰ. DeepLebV3  
Ⅱ. Co-attention  
Ⅲ. Segmentation  
#### DeepLabV3
Pretrained weighted DeepLabV3 model

    https://drive.google.com/file/d/1hy0-BAEestT9H4a3Sv78xrHrzmZga9mj/view

Traditional DeepLabv3 architecture consist of ResNet101 and [Atrous Spatial Pyramid Pooling](https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d).However, because of limited of GPU memory, ResNet101 is switched into ResNet50.  
##### Atrous Spatial Pyramid Pooling  
Atrous Spatial Pyramid Pooling is actually Atrous Dilated Convolution. In DeepLabV3, it use dilated rate [6,12,18] to process feature maps, extracting the features containing more useful information. After combining them all together, ASPP creates the final global feature. In this case, ASPP offers more information and higher accuracy   
#### Co-attention
![image](https://github.com/FangLintao/Video-Object-Segmentation/blob/master/images/3.png)    
###### reference: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks,Xiankai Lu,Wenguan Wang,Chao Ma,Jianbing Shen,Ling Shao,Fatih Porikli
![image](https://github.com/FangLintao/Video-Object-Segmentation/blob/master/images/4.png)  
![image](https://github.com/FangLintao/Video-Object-Segmentation/blob/master/images/5.png)  
#### Segmentation  

    Ⅰ. two 3*3 convolution layers   
    Ⅱ. one 1*1 convolution layer 

## Implementation
### Device Requirement
Ⅰ. GPU 8G  
Ⅱ. Pytorch

        For Training  
        --  TIANCHI-Training.py  
        For Testing  
        --  TIANCHI-Testing.py

## Reference
Ⅰ. Unsupervised Video Object Segmentation with Co-Attention Siamese Networks,Xiankai Lu,Wenguan Wang,Chao Ma,Jianbing Shen,Ling Shao,Fatih Porikli, Inception Institute of Artificial Intelligence, UAE
