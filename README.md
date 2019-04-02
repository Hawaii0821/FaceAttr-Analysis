# FaceAttr-Analysis
This repo is for my adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## File Description

| File/Folder | Description |
| ----------- | ----------- |
| \paper | This folder keeps papers relevant to face attibutes analysis.|
| CelebA.py | This file defines the dataset class for CelebA and provides the data loader function. |
| FaceAttr_baseline_model.py | This file offers the baseline model class, consisting of feature extraction submodel (resnet etc.) and feature classfier submodel (full connect)|
|analysis_attr.py | It reflects the relationship between positive samples and negetive samples in CelebA.|
|solver.py|The file has many functions like initializing, training and evaluating model.|
|main.py| The entry file of project that owns some important variables.|
| logger.py | Use tensorboardX for visualization. |
| sample_num.csv | It records the number of positive and negative samples on every attribute.|

## Dependency
> pip install -r requirements.txt 

## Todo
- [ ] Visualization with tensorboard.
- [ ] Try more famous models, such as ResNet50, ResNet101, DenseNet, ResNeXt, SENet.
- [ ] Customize the network structure.
- [ ] Parse the input script command. 
- [ ] Search for the appropriate prediction threshold for every attribute or find a good place to teach themselves.
- [ ] Front end: Video stream monitor[(picamera)](https://github.com/waveform80/picamera) and transfer video frames.
- [ ] Back end: [face detection](https://github.com/ageitgey/face_recognition) and real-time analysis. 

## Done
- [x] Attribute analysis.
- [x] Data loader.
- [x] Built baseline model(Resnet18).
- [x] Train and evaluate of multiple tasks. 
- [x] Save and load model.

# Related Resource
* [multi-task-learning](https://paperswithcode.com/task/multi-task-learning)
* [image classification](https://paperswithcode.com/task/image-classification)


## License
[MIT](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/LICENSE).
