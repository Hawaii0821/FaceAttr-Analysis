# FaceAttr-Analysis
This repo is for my adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## File Description

| File/Folder | Description |
| ----------- | ----------- |
| \paper | This folder keeps the papers relevant to face attibutes analysis.|
| CelebA.py | This file defines the dataset class for CelebA and provides the data loader function. |
| FaceAttr_baseline_model.py | This file offers the baseline model class, consisting of feature extraction submodel (resnet etc.) and feature classfier submodel (full connect)|
|analysis_attr.py | It reflects the relationship between positive samples and negetive samples in CelebA.|
|solver.py|The file has many functions like initializing, training and evaluating model.|
|main.py| The entry file of project that owns some important variables.|
| logger.py | Use tensorboardX for visualization. |

## Dependency
> pip install -r requirements.txt 

## Todo
-[] Visualization with tensorboard.
-[] Try more famous models, such as ResNet50, ResNet101, DenseNe, ResNeXt, SENet.
-[] Video stream monitor and real-time analysis.
-[] Customize the network structure.
-[] Parse the input script command.
-[] Search for the appropriate prediction threshold for each attribute.
