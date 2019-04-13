# FaceAttr-Analysis

This repo is for the adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## File Description

| File/Folder | Description |
| ----------- | ----------- |
|**Folders**||
| \paper | This folder keeps papers relevant to face attibutes analysis.|
| \model | The trained model and the evaluatiing result including model dict, loss and accuracy csv files. |
|\front-back-end| the front end html to upload image and the backend flask to receive image.|
|**Main files**||
| CelebA.py | This file defines the dataset class for CelebA and provides the data loader function. |
| FaceAttr_baseline_model.py | This file offers the baseline model class, consisting of feature extraction submodel (resnet etc.) and feature classfier submodel (full connect)|
|analysis_attr.py | It reflects the relationship between positive samples and negetive samples in CelebA.|
|solver.py|The file has many functions like initializing, training and evaluating model.|
|main.py| The entry file of project that owns some important variables.|
| logger.py | Use tensorboardX for visualization. |
|camera.py| Open camera and detect face.|
|haarcascade_frontalface_default.xml| The model dict of detecting face with opencv.|
|**Recoding files**||
| sample_num.csv | It records the number of positive and negative samples on every attribute.|
|sample_num.png| It shows the distribution of attributes.|
| DailyNote.md | The recording note of this project.|
| requirements.txt | The requirements file which save the needed package info. |  

## TODO

- [ ] Visualization with [tensorboard](https://github.com/lanpa/tensorboardX) or [netron](https://github.com/lutzroeder/netron).
- [ ] Try more famous models, such as ResNet50, ResNet101, DenseNet, ResNeXt, SENet.
- [ ] Customize the network structure.
- [ ] Parse the input script command.
- [ ] Search for the appropriate prediction threshold for every attribute or find a good place to teach themselves.
- [x] Front end: Video stream monitor[(picamera)](https://github.com/waveform80/picamera)(flask) and transfer video frames.
- [ ] Back end: [face detection](https://github.com/ageitgey/face_recognition) and real-time analysis.
- [x] [Attribute analysis](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/analysis_attr.py).
- [x] [Data process and load](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/CelebA.py).
- [x] [Built baseline model(Resnet18 and 101)](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/FaceAttr_baseline_model.py).
- [x] [Train and evaluate of multi-task](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/solver.py).
- [x] Save and load model.

## Problems
- Sample attributes imbalance.
- Need more effective learning strategys on multi-label task. (loss function? network structure?...)


## Related Resource

- [multi-task-learning](https://paperswithcode.com/task/multi-task-learning)
- [image classification](https://paperswithcode.com/task/image-classification)
- [LFW Face DataBase](http://vis-www.cs.umass.edu/lfw/)

## Dependency & OS

> pip install -r requirements.txt   # created by cmd: pipreqs ./

> linux 16.0.1

## License

[MIT](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/LICENSE).
