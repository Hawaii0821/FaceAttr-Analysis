# FaceAttr-Analysis

This repo is for the adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## Plan

### Dataset

- CelebA: about 162770 train images and  about 39829 test images. （with precropped images, but it's better to try wild images.）
- LFWA: todo....(Can we train a model on different datasets? Yeah)

### Methods

- [x] v1:resnet101 + self-defined fc + BCE loss with logits + average accuracy
- [x] v2:resnet101 + self-defined fc + BCE loss with logits + threshold rescaling (`from page 67,Machine Learning, Zhihua Zhou`) + average accracy + more metrics (recall, precision, TPR, FPR, F1 `from page 30-33,Machine Learning, Zhihua Zhou`)
- [x] v3:GC_resnet101
- [ ] Other: SENet ....

## Experiment Result

### Our Work

| plan | avearage accuracy(%)| macro-precision(%) | macro-recall(%) | macro-F1(%) |
| ---- | -----| ---- | ----- | ----- |
| Resnet101-v1  |  91.14 |--- | ---| ---|
| Resnet101-v2 | 90.07 | 0.67 | 0.18 | 0.28 |
| GC_resnet101-v3 | 89.06| 0.68|0.17 | 0.27 |

More detailed data can be seen in folder [\model](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/model/).

### State of the Art

![State of the art](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/celeba.png)
> The image is from this paper--[FAN,Youtu Search,](https://www.ijcai.org/proceedings/2018/102)

## Simple Dashbord

Priority:

- [ ] Try more famous models, such as ResNet101, DenseNet, ResNeXt, SENet.(in processing)
- [ ] Customize the network structure for better performance.
- [ ] Open camera of laptop and real-time analyis. （in processing)
- [ ] Search for the appropriate prediction threshold for every attribute or find a good place to teach themselves.
- [ ] More detailed analysis about the experiment results.(in processing)
- [ ] Train on wild images instead of precroped images to get a higher performance in practice.
- [ ] Face localization,alignment and analysis.(challenging)
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
- [x] [Train and evaluate](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/solver.py).
- [x] Save and load model.

More study notes on the [DailyNote.md](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/DailyNote.md)

## Problems

- Sample attributes imbalance.
- Need more effective learning strategies on multi-label task. (loss function? network structure?...)
- The wild images differ from training and test image (aligned)

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
