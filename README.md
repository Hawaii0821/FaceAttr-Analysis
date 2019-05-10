# FaceAttr-Analysis

This repo is for the adavanced training on deeping learning with the purpose of building a face attributes analysis application.

## Plan

### Dataset

- CelebA: about 162770 train images and  about 39829 test images. （with precropped images, but it's better to try wild images.）
- LFWA: todo....(Can we train a model on different dataset? Yeah) 

### Methods

- [x] v1:resnet101 + self-defined fc + BCE loss with logits + average accuracy
- [x] v2:resnet101 + self-defined fc + BCE loss with logits + threshold rescaling (`from page 67,Machine Learning, Zhihua Zhou`) + average accracy + more metrics (recall, precision, TPR, FPR, F1 `from page 30-33,Machine Learning, Zhihua Zhou`)
- [ ] v3:Based on v1, try to adopt focal loss or OHEM. (to reduce the effect of samples imbalance)
- [ ] Other: SENet + tanh + hinge loss....

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

## File Description
<details>
<summary> Click for details. </summary>
<pre><code>

<table border='2' cellpadding='5' cellspacing='0'>
  <tr>
    <td>File/Folder</td>
    <td>Description</td>
  </tr>
  <tr>
    <td><b>Folders</b></td>
    <td>-----------------------------------------------------------------------</td>
  </tr>
  <tr>
    <td>\model</td>
    <td>The trained model and the evaluatiing result including model dict, loss and accuracy csv files.</td>
  </tr>
  <tr>
    <td>\paper</td>
    <td>This folder keeps papers relevant to face attibutes analysis.</td>
  </tr>
  <tr>
    <td>\front-back-end</td>
    <td>the front end(html) to upload images and the back end(flask) to receive images.</td>
  </tr>
  <tr>
    <td><b>Main files</b></td>
    <td>------------------------------------------------------------------------</td>
  </tr>
  <tr>
    <td>CelebA.py</td>
    <td>This file defines the dataset class for CelebA and provides the data loader function. </td>
  </tr>
  <tr>
    <td>FaceAttr_baseline_model.py</td>
    <td>This file offers the baseline model class, consisting of feature extraction submodel (resnet etc.) and feature classfier submodel (full connect)</td>
  </tr>
  <tr>
    <td>analysis_attr.py</td>
    <td>It reflects the relationship between positive samples and negetive samples in CelebA.</td>
  </tr>
  <tr>
    <td>solver.py</td>
    <td>The file has many functions like initializing, training and evaluating model.</td>
  </tr>
  <tr>
    <td>main.py</td>
    <td>The entry file of project that owns some important variables.</td>
  </tr>
  <tr>
    <td>camera.py</td>
    <td>Open camera and detect face.</td>
  </tr>
  <tr>
    <td>haarcascade_frontalface_default.xml</td>
    <td>The model dict of detecting face with opencv.</td>
  </tr>
 <tr>
    <td>exp_resylt_analysis.py</td>
    <td>Show the experiment result and other visual application</td>
  </tr>
</table>
</details>

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
