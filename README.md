# FaceAttr-Analysis

This repo is for the adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## Plan

### Dataset

- CelebA: about 162770 train images and  about 39829 test images. （with precropped images, but it's better to try wild images.）
- LFWA: todo....(Can we train a model on different datasets? Yeah)

### Methods

- [x] v1:resnet101 + self-defined fc + BCE loss with logits + average accuracy
- [x] v2:resnet101 + self-defined fc + BCE loss with logits + threshold rescaling (`from page 67,Machine Learning, Zhihua Zhou`) + average accracy + more metrics (recall, precision, TPR, FPR, F1 `from page 30-33,Machine Learning, Zhihua Zhou`)
- [x] v3+v5: try GC_resnet101 after modifying label format and loss. They can be finetuned to be better.
- [x] v4: resnet101 + focal loss.
- [x] v6: try SENet
- [x] v7: resnet152
- [ ] v8: FAN(pix2pix/unet + dual-path network) from youtu search. Move to another [repo](https://github.com/JoshuaQYH/pytorch.FAN) to check.

## Experiment Result

### Our Work

| plan | avearage accuracy(%)| macro-precision(%) | macro-recall(%) | macro-F1(%) | speed(pictures/s)| comment |
| ---- | -----| ---- | ----- | ----- | --- |  ---- |
| Resnet101-v1  |  91.14 |--- | ---| ---| ---| test on eval&test dataset--deprecated|
| Resnet101-v2 | 90.07 | 67.46 | 68.18 | 67.35 | ---| test on eval& test dataset--deprecated.|
| GC_resnet101-v3 | 89.06| 55.06| 64.19 | 58.47 |---|test on test dataset |
| Resnet101-v4 | 84.35| 59.97| 40.39| 53.06 |---|test on test dataset|  
| GC_resnet101-v5 | 89.93| 77.18 |47.67 |57.40|---|test on test dataset|
| SE_resnet101-v6 | 90.06 | 77.35 | 49.67 | 59.42 |---  | test on test dataset|
| Resnet152-v7 | 88.72 | 74.64 | 43.81 | 54.78 | --- |

More detailed data can be seen in folder [\result](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/result/).

### State of the Art

![State of the art](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/celeba.png)
> The image is from this paper--[FAN,Youtu Search,](https://www.ijcai.org/proceedings/2018/102)

## Simple Dashbord

Priority:

- [ ] Try more famous models, such as ~~ResNet101~~, DenseNet, ResNeXt, ~~SENet~~, ~~GCNet~~.(in processing)
- [ ] Customize the network structure for better performance.
- [ ] Open camera of laptop and real-time analyis.
- [ ] Train a model on different datasets for more attributes.
- [ ] upload image in html and return the analysis result.
- [ ] search face attributes and return the related images.
- [ ] open pc camera, detect face and return the predicted result.

Choice:

- [ ] Visualization with [tensorboard](https://github.com/lanpa/tensorboardX) or [netron](https://github.com/lutzroeder/netron).
- [ ] Train on wild images instead of precroped images to get a higher performance in practice.
- [ ] Face localization, alignment and analysis. (challenging)

Done:

- [x] [Dataset attribute analysis](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/analysis_attr.py).
- [x] [Data process and load](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/CelebA.py).
- [x] [Built baseline model(Resnet18 and 101)](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/FaceAttr_baseline_model.py).
- [x] [Train and evaluate](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/solver.py).
- [x] Save and load model.
- [x] More detailed analysis about the experiment results.
- [x] Parse the input script command.
- [ ] 
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
