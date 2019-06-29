# Daily Notes

Some notes, summary or ideas on the process of learning with unfixed format in every week.

## Week15 ~ 17

- build the web app with face detection API and face analysis model based on django.
- try other models: cbam_resnet, shufflenetv2...

## Week14

(19.05.27~19.06.02)

* train the best model on resnet152, gc_resnet101.
* train the new and better network: sknet, sgenet.

## Week13

(19.05.20~05.26)

* more data agumentation methods.
* test speed function.
* finetune existing models(gc_resnet101, resnet101, senet101)
* read papers related to face attribute analysis.
* to rebuild the state of the art methods FAN network.

## Week12

 (19.05.15)

* try the focal loss. (non-ideal...)
* try GCNet
* try senet
* fix bugs

## Week11

(19.05.11)

### Improvement and discussion

- More data visualization on process and result
- spilit dataset into validate and test dataset.
- focal loss or OHEM to solve sample imbalance.
- use the wild images(?).
- find a tradeoff between speed and accuracy.
- multi-task: face localization and analysis.

## Week9

(19.04.27 ~ 04.29)

### Experiment summary

Add threshold scaling, causing worse accuracy (resnet v2). Maybe we should define some thresholds for some worse attributes by observing the confusion matrix.

By observing the accuracy file, we found that the attributes whose accuracy is under 0.85 are:

|Attributes| Accuracy| Reason|
|------|-----|-------|
|2. Arched_Eyebrows：柳叶眉 | 83.17%| +|
|3. Attractive：吸引人的 | 81.77% | - |
|4. Bags_Under_Eyes：眼袋|82.51%| + |
|7. Big_Lips：大嘴唇|75.12% |+ |
| 8. Big_Nose：大鼻子 | 82.60%|+ |
|26. Oval_Face：椭圆形的脸| 75.04%| + |
| 28. Pointy_Nose：尖鼻子 | 73.85% | + |
| 33. Straight_Hair：直发 | 82.73% | + |
| 38. Wearing_Necklace：戴着项链 | 81.56% | o |

`+` means imbalanced samples, `-` means unmearsuable attributes, `o` means uncertained.

The attributes whose accuracy is above 0.98 are:

|Attributes| Accuracy| Reason|
|------|----|-----|
| 5. Bald：秃头 | 98.07% | # |
|16. Eyeglasses：眼镜 | 99.49% | # |
| 21. Male：男性 | 98.48%| ! |
| 36. Wearing_Hat：戴着帽子 | 98.90% | # |

`#` means imbalanced samples but obvious attributes. `!` means balanced sample.

### Some ideas

- finish the adaptive attribute thereshold and weight loss task.
- compare result with the state of art or some other baseline.
- train on raw image instead of aligned image.(optional)
- try to record other evaluation metric, like recall, precision and balanced accuracy.
- more adversarial Robustness
- tanh + hinge
- focal loss

### Paper

- Semantic segmentation to help analysis.
- Harnessing Synthesized Abstraction Images to Improve Facial Attribute Recognition.(Youtu)
- Facial Attributes: Accuracy and Adversarial Robustness. （Transfomer)
- Multi-task: face detection and face analysis.

## Week7 

(19.04.08)

- [x] Open camera and capture video frame by opencv3.
- [x] Upload the picture on front end and receive it at backend. (Flask)
- [x] Use haarcascades detector on opencv to detect face. （Fast but not accurate.)

## Week6 

(19.04.01)

### Some goals

- find a adaptive and good method to adjuest attribute classification threshold and attribute loss weight.
- try to find a way to train on different datasets, such as CelebA, LFW. (The mask vector lable from StarGAN)
- use picamera to build a simple to transfer video frame.
- use face_regonition to extract faces.
- adopt more meaningful metrics (precision or recall..) instead of accuracy.

### New difficulity

- the trainning samples content is quite different from the detected face images.

## Week5 

(19.03.30 ~ 19.03.31)

### Difficulty & Solution

- multi-task learning & multi-label cls
  - gradient domination: add weight for each task loss or cost matrix.
  - loss function: sigmoid + F. binary_cross_entropy_with_logits
- samples issue:
  - samples imbalance: undersampling; oversampling; threshold-moving/rescaling.
  - Feature is not obvious and hard to detect, such as the attribute 'attractive'. Semantic segmentation ?

## Week4 

(19.03.23 ~ 19.03.24)

### Paper Note

#### Opportunity

- Attributes are especially useful in modeling intra-category variations such as fine-grained classification.
- Face retrieval.
- Intelligent retail on Big Data of customer styles.
- Face recognition.

#### Challenge

- Hard to define a spatial boundary for a given attribute.
- Find an auxiliary task to find detailed localization information without restricting the corresponding regions to be in certain pre-defined shapes.
- Multiple classes classification.

#### Possible Solution

- Semantic segmentation to offer facial attributes localization cues in the form of semantic segmentation that decompose the spatial domain of an image into mutally exclusive semantic regions.
- Try to leverages facial parts locations for better attribute prediction. Generate a facial abstraction image which contains both local facial parts and facial texture information.

#### Paper list

1. Improving Facial Attribute Prediction using Semantic Segmentation. [Arxiv](https://arxiv.org/abs/1704.08740)
2. Harnessing Synthesized Abstraction Images to Improve Facial Attribute Recognition. [IJCAI](https://www.ijcai.org/proceedings/2018/102)

## Week3 

(19.03.16)

- A simple analysis note in `CelebA_analysis.md`
- A simple visualzation of every attributes' sample numbers in `analysis_attr.py`.
