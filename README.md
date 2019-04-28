# FaceAttr-Analysis

This repo is for the adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## Planning Model

- [x] resnet101 + self-defined fc + sigmoid + BCE loss with logits
- [] ....  

<<<<<<< HEAD
## Experiment Result



=======
>>>>>>> 6df6046d450b96e534e163b1c60236242862fb90
## File Description

| File/Folder | Description |
| ----------- | ----------- |
|**Folders**||
| \paper | This folder keeps papers relevant to face attibutes analysis.|
| \model | The trained model and the evaluatiing result including model dict, loss and accuracy csv files. |
|\front-back-end| the front end(html) to upload images and the back end(flask) to receive images.|
|**Main files**||
| CelebA.py | This file defines the dataset class for CelebA and provides the data loader function. |
| FaceAttr_baseline_model.py | This file offers the baseline model class, consisting of feature extraction submodel (resnet etc.) and feature classfier submodel (full connect)|
|analysis_attr.py | It reflects the relationship between positive samples and negetive samples in CelebA.|
|solver.py|The file has many functions like initializing, training and evaluating model.|
|main.py| The entry file of project that owns some important variables.|
| logger.py | Use tensorboardX for visualization. |
|camera.py| Open camera and detect face.|
|haarcascade_frontalface_default.xml| The model dict of detecting face with opencv.|
|exp_resylt_analysis.py|Show the csv result and other visual application|
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
- [ ] Plan1: video stream monitor[(picamera on Raspberry Pi)](https://github.com/waveform80/picamera) and transfer video frames.
- [ ] Plan2: upload image in html and return the analysis result.
- [ ] Plan3: open camera of laptop and real-time analyis.
- [ ] Back end: [face detection](https://github.com/ageitgey/face_recognition) and real-time analysis.
- [x] [Attribute analysis](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/analysis_attr.py).
- [x] [Data process and load](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/CelebA.py).
- [x] [Built baseline model(Resnet18 and 101)](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/FaceAttr_baseline_model.py).
- [x] [Train and evaluate of multi-task](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/solver.py).
- [x] Save and load model.

More study notes on the [DailyNote.md](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/DailyNote.md)

## Problems

- Sample attributes imbalance.
- Need more effective learning strategies on multi-label task. (loss function? network structure?...)
- The wild images differ from training and test image (aligned)

## Related Resource

- [multi-task-learning](https://paperswithcode.com/task/multi-task-learning)
- [image classification](https://paperswithcode.com/task/image-classification)
- [LFW Face DataBase](http://vis-www.cs.umass.edu/lfw/)

## Dependency & OS

<<<<<<< HEAD
> pip install -r requirements.txt   // requirements.txt created by cmd: pipreqs ./
=======
> pip install -r requirements.txt   # created by cmd: pipreqs ./

>>>>>>> d660a031171a2501490b672b3862c14b9547c978
> linux 16.0.1

## License

[MIT](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/LICENSE).
