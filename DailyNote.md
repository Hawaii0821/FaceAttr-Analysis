# Notes

## Week7 (19.04.08)

- [*] Open camera and capture video frame by opencv3.
- [*] Upload the picture on front end and receive it at backend. (Flask)
- [*] Use haarcascades detector on opencv to detect face. （Fast but not accurate.)

## Week6 (19.04.01)

### Some goals:

- find a adaptive and good method to adjuest attribute classification threshold and attribute loss weight.
- try to find a way to train on different datasets, such as CelebA, LFW. (The mask vector lable from StarGAN)
- use picamera to build a simple to transfer video frame.
- use face_regonition to extract faces.
- adopt more meaningful metrics (precision or recall..) instead of accuracy.

### New difficulity:

- the trainning samples content is quite different from the detected face images.

## Week5 (19.03.30 ~ 19.03.31)

### Difficulty & Solution

- multi-task learning & multi-label cls
  - gradient domination: add weight for each task loss or cost matrix.
  - loss function: sigmoid + F. binary_cross_entropy_with_logits
- samples issue:
  - samples imbalance: undersampling; oversampling; threshold-moving/rescaling.
  - Feature is not obvious and hard to detect, such as the attribute 'attractive'. Semantic segmentation ?

## Week4 (19.03.23 ~ 19.03.24)

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
