# FaceAttr-Analysis

This repo is for the adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## Plan

### Dataset

- CelebA: about 162770 train images and  about 39829 test images.
- LFWA: todo....

### Methods

- [x] v1:resnet101 + self-defined fc + BCE loss with logits + average accuracy
- [x] v2:resnet101 + self-defined fc + BCE loss with logits + threshold rescaling (`from page 67,Machine Learning, Zhihua Zhou`) + average accracy + more metrics (recall, precision, TPR, FPR, F1 `from page 30-33,Machine Learning, Zhihua Zhou`)
- [ ] v3:SENet + tanh + hinge + loss....

## Experiment Result

### Our Work

| plan | avearage accuracy(%)| macro-precision(%) | macro-recall(%) | macro-F1(%) |
| ---- | -----| ---- | ----- | ----- |
| Resnet101-v1  |  91.14 |--- | ---| ---|
| Resnet101-v2 | 90.07 | 0.67 | 0.68 | 0.67 |

More detailed data can be seen in folder [\model](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/model/).

### State of the Art

![State of the art](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/celeba.png)
> The image is from this paper--[FAN,Youtu Search,](https://www.ijcai.org/proceedings/2018/102)

## Simple Dashbord

Priority:

- [ ] Try more famous models, such as ResNet50, ResNet101, DenseNet, ResNeXt, SENet.
- [ ] Customize the network structure for better performance.
- [ ] Open camera of laptop and real-time analyis. 
- [ ] Search for the appropriate prediction threshold for every attribute or find a good place to teach themselves.
- [ ] More detailed analysis about the experiment results.
- [ ] Train on wild images instead of precroped images to get a higher performance in practice.
- [ ] Face localization and alignment.
- [ ] Train a model on different datasets by processing labels.

Choice:

- [ ] Visualization with [tensorboard](https://github.com/lanpa/tensorboardX) or [netron](https://github.com/lutzroeder/netron).
- [ ] Parse the input script command.
- [ ] video stream monitor[(picamera on Raspberry Pi)](https://github.com/waveform80/picamera) and transfer video frames.
- [ ] upload image in html and return the analysis result.
- [ ] Back end: [face detection](https://github.com/ageitgey/face_recognition) and real-time analysis.
- [ ] Create adversarial samples for robustness.

Done

- [x] [Dataset attribute analysis](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/analysis_attr.py).
- [x] [Data process and load](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/CelebA.py).
- [x] [Built baseline model(Resnet18 and 101)](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/FaceAttr_baseline_model.py).
- [x] [Train and evaluate of multi-task](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/solver.py).
- [x] Save and load model.

More study notes on the [DailyNote.md](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/DailyNote.md)

## Problems

- Sample attributes imbalance.
- Need more effective learning strategies on multi-label task. (loss function? network structure?...)
- The wild images differ from training and test image (aligned)

## File Description

| File/Folder | Description |
| ----------- | ----------- |
|**Folders**|--------------------------------------------------------------------------------------|
| \paper | This folder keeps papers relevant to face attibutes analysis.|
| \model | The trained model and the evaluatiing result including model dict, loss and accuracy csv files. |
|\front-back-end| the front end(html) to upload images and the back end(flask) to receive images.|
|**Main files**|-----------------------------------------------------------------------------------|
| CelebA.py | This file defines the dataset class for CelebA and provides the data loader function. |
| FaceAttr_baseline_model.py | This file offers the baseline model class, consisting of feature extraction submodel (resnet etc.) and feature classfier submodel (full connect)|
|analysis_attr.py | It reflects the relationship between positive samples and negetive samples in CelebA.|
|solver.py|The file has many functions like initializing, training and evaluating model.|
|main.py| The entry file of project that owns some important variables.|
| logger.py | Use tensorboardX for visualization. |
|camera.py| Open camera and detect face.|
|haarcascade_frontalface_default.xml| The model dict of detecting face with opencv.|
|exp_resylt_analysis.py|Show the experiment result and other visual application|
|**Recoding files**|-----------------------------------------------------------------------|
| sample_num.csv | It records the number of positive and negative samples on every attribute.|
|sample_num.png| It shows the distribution of attributes.|
| DailyNote.md | The recording note of this project.|
| requirements.txt | The requirements file which save the needed package info. |  

## Related Resource

- [multi-task-learning](https://paperswithcode.com/task/multi-task-learning)
- [image classification](https://paperswithcode.com/task/image-classification)
- [LFWA Face Attribute DataBase](http://vis-www.cs.umass.edu/lfw/)

## Dependency & OS

> pip install -r requirements.txt   # requirements.txt created by cmd: pipreqs ./
> 
> linux 16.0.1

## License

[MIT](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/LICENSE).
