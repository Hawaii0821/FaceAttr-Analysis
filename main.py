from solver import Solver


# ----------- model/train/test configuration ---- #
epoches = 3

batch_size = 64

learning_rate = 0.0001

model_type = "Resnet50"

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

attr_nums = [1, 2, 3,4,5,] 
attr_loss_weight = [1, 0.8, 0.7]  
attr_threshold = [0.6, 0.5, 0.7] 
selected_attrs = []
for num in attr_nums:
    selected_attrs.append(num)


# ------------- Path setting --------------------- #

log_dir = "./log"
# You should download the celeba dataset in the root dir.
image_dir = "../CelebA/Img/img_align_celeba/" 
attr_path = "../CelebA/Anno/list_attr_celeba.txt"

# -------------- Tensorboard --------------------- #
use_tensorboard = False


#--------------- exe ----------------------------- # 
if __name__ == "__main__":
    solver = Solver(epoches=epoches, batch_size=batch_size, learning_rate=learning_rate,
                    model_type=model_type, optim_type=optim_type, momentum=momentum, pretrained=pretrained,
                    selected_attrs=selected_attrs, image_dir = image_dir, attr_path=attr_path, log_dir = log_dir, 
                    use_tensorboard = use_tensorboard,attr_loss_weight=attr_loss_weight, attr_threshold=attr_threshold)
    solver.fit()
