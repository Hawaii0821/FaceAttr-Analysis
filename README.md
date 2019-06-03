# FaceAttr-Analysis

This repo is for the adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## Plan

### Dataset

- CelebA: about 162770 train images and  about 39829 test images. （with precropped images, but it's better to try wild images.）
- LFWA: (todo...because the original dataset link is invalid...)  

### Methods

- [x] v1:resnet101 (self-defined fc + BCE loss with logits + average accuracy)
- [x] v2:resnet101  (self-defined fc + BCE loss with logits + threshold rescaling (`from page 67,Machine Learning, Zhihua Zhou`) + average accracy + more metrics (recall, precision, TPR, FPR, F1 `from page 30-33,Machine Learning, Zhihua Zhou`))
- [x] v3+v5: try GC_resnet101 after modifying label format and loss. 
- [x] v4: resnet101 + focal loss.
- [x] v6: se_resnet101
- [x] v7: resnet152
- [x] v8: densenet121
- [x] v9: [SGE_resnet101](https://arxiv.org/pdf/1905.09646.pdf).
- [x] V10: [SK_resnet101](https://arxiv.org/pdf/1903.06586.pdf).
- [ ] v11: FAN(pix2pix/unet + dual-path network) from youtu search. Move to another [repo](https://github.com/JoshuaQYH/pytorch.FAN) to check.

## Experiment Result

### Our Work

| plan | avearage accuracy(%)| macro-precision(%) | macro-recall(%) | macro-F1(%) | speed(pictures/s)| comment |
| ---- | -----| ---- | ----- | ----- | --- |  ---- |
| Resnet101-v1  |  91.14 |--- | ---| ---| ---|test on val&test dataset|
| Resnet101-v2 | 91.53 | 79.81 | 63.67 | 68.52 | ---| test dataset|
| GC_resnet101-v3 | 89.06| 55.06| 64.19 | 58.47 |---| test&val dataset |
| Resnet101-v4 | 84.35| 59.97| 40.39| 53.06 |---| test dataset(deprecated)|  
| GC_resnet101-v5.2 | 91.94| 79.45 |65.64 |69.94|---| a finetuned version on test dataset|
| SE_resnet101-v6.2 | 91.95 | 79.45 | 65.64 | 69.99 |---  |test dataset|
| Resnet152-v7.1 | 91.95 | 79.46 | 65.98 | 70.14 | --- | test dataset|
| Densenet121-v8| 91.60 | 79.23 | 65.40 | 69.77 | --- | test dataset |
| SGE_resnet101-v9 | 91.60 | 79.23 | 65.40 | 69.77| ---|test dataset |
| SK_resnet101-v10 | 91.93 | 79.69 | 65.54 | 69.95| -- | test dataset|

More detailed data can be seen in folder [\result](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/result/).

### State of the Art

![State of the art](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/celeba.png)
> The image is from this paper--[FAN,Youtu Search,](https://www.ijcai.org/proceedings/2018/102)

## Simple Dashbord

Priority:

- [x] Try more famous models, such as ~~ResNet101~~, ~~DenseNet~~, ~~SKNet~~, ~~SGENet~~, ~~SENet~~, ~~GCNet~~.(in processing)
- [x] Customize the network structure for better performance.
- [ ] upload images in html and return the analysis result.
- [ ] open pc camera, detect face and return the predicted result.
- [ ] search face attributes and return the related images.

Choice:

- [ ] Visualization with [tensorboard](https://github.com/lanpa/tensorboardX) or [netron](https://github.com/lutzroeder/netron).
- [ ] Train on wild images instead of precroped images to get a higher performance in practice.
- [ ] Multi-task learning: face localization, alignment and analysis. (challenging)

Done:

- [x] [Dataset attribute analysis](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/analysis_attr.py).
- [x] [Data process and load](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/CelebA.py).
- [x] [Built baseline model(Resnet18 and 101)](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/FaceAttr_baseline_model.py).
- [x] [Train and evaluate](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/solver.py).
- [x] Save and load model.
- [x] More detailed analysis about the experiment results.
- [x] Parse the input script command.
More study notes on the [DailyNote.md](https://github.com/JoshuaQYH/FaceAttr-Analysis/blob/master/DailyNote.md)

## Usage

Open the `run.sh` file, read the main.py and you can see the running arguments of model.
> bash run.sh

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
