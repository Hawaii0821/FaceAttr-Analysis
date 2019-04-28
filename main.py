from solver import Solver
import os
import random
import numpy as np 
import torch
import pandas as pd 
# ------------- Path setting --------------------- #

log_dir = "./log"
# You should download the celeba dataset in the root dir.

# the dataset local path.
# image_dir = "../CelebA/Img/img_align_celeba/" 
# attr_path = "../CelebA/Anno/list_attr_celeba.txt"

#the dataset path run on server.
image_dir = "../../dataset/CelebA/Img/img_align_celeba/" 
attr_path = "../../dataset/CelebA/Anno/list_attr_celeba.txt"

# ----------- model/train/test configuration ---- #
epoches = 50  # 50

batch_size = 128

learning_rate = 0.0001

model_type = "Resnet101"  # 34 50 101 152

optim_type = "SGD"

momentum = 0.9

pretrained = True



# -------------- Attribute configuration --------- #

# every row has 5 attributes.
all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
            'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
            'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' 
]

# To be optimized
attr_nums = [i for i in range(len(all_attrs))] 
attr_loss_weight = [1 for i in range(len(all_attrs))]  

selected_attrs = []
for num in attr_nums:
    selected_attrs.append(all_attrs[num])

# To solve the sample imbalance called rescaling, If the threshold > m+ /(m+ + m-), treat it as a positive sample. 
attr_threshold = [0.5 for i in range(len(all_attrs))]  

""" Cause worse accuracy result.
sample_csv = pd.read_csv('sample_num.csv')
attr_threshold = (sample_csv['positive sample']/(sample_csv['positive sample'] + sample_csv['negative sample'])).tolist()
"""

# -------------- Tensorboard --------------------- #
use_tensorboard = False


# ------------------------------------------------ # 
# make sure the same results with same params in different running time.
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


#--------------- exe ----------------------------- # 
if __name__ == "__main__":
    solver = Solver(epoches=epoches, batch_size=batch_size, learning_rate=learning_rate,
                    model_type=model_type, optim_type=optim_type, momentum=momentum, pretrained=pretrained,
                    selected_attrs=selected_attrs, image_dir = image_dir, attr_path=attr_path, log_dir = log_dir, 
                    use_tensorboard = use_tensorboard,attr_loss_weight=attr_loss_weight, attr_threshold=attr_threshold)
    solver.fit()
