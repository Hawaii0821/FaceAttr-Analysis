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

## Analysis of the task
### Opportunity
* Attributes are especially  useful in modeling intra-category variations such as fine-grained classification.
* Face retrieval.
* Intelligent retail on Big Data of customer styles.
* Face recognition.
### Challenge
* Hard to define a spatial boundary for a given attribute.
* Find an auxiliary task to find detailed localization information without restricting the corresponding regions to be in certain pre-defined shapes.
* Multiple classes classification.
### Possible Solution
* Semantic segmentation to offer facial attributes localization cues in the form of semantic segmentation that decompose the spatial domain of an image into mutally exclusive semantic regions.
* Try to leverages facial parts locations for better attribute prediction. Generate a facial abstraction image which contains both local facial parts and facial texture information. . 
