> Some simple notes in every training weeek.



# Week5 (19.03.30 ~ 19.03.31)
## Difficulty & Solution
* multi-task learning
  * sample imbalance: resample; 
  * gradient domination: add weight for each task loss or cost matrix.
* multi-label:
  * sigmoid + F. binary_cross_entropy_with_logits

# Week4 (19.03.23 ~ 19.03.24)
## Paper Note
### Opportunity
* Attributes are especially  useful in modeling intra-category variations such as fine-grained classification.
* Face retrieval.
* Intelligent retail on Big Data of customer styles.
* Face recognition.
### Challenge
* Hard to define a spatial boundary for a given attribute.
* Find an auxiliary task to find detailed localization information without restricting the corresponding regions to be in certain pre-defined shapes.
* Multiple classes classification.
### Possible Solution
* Semantic segmentation to offer facial attributes localization cues in the form of semantic segmentation that decompose the spatial domain of an image into mutally exclusive semantic regions.
* Try to leverages facial parts locations for better attribute prediction. Generate a facial abstraction image which contains both local facial parts and facial texture information. 

### Paper list
1. Improving Facial Attribute Prediction using Semantic Segmentation. [Arxiv](https://arxiv.org/abs/1704.08740)
2. Harnessing Synthesized Abstraction Images to Improve Facial Attribute Recognition. [IJCAI](https://www.ijcai.org/proceedings/2018/102)
