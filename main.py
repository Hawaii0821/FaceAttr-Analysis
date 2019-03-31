from solver import Solver


# ----------- model/train/test configuration ---- #
epoches = 3

batch_size = 64

learning_rate = 0.0001

model_type = "Resnet18"

optim_type = "SGD"

momentum = 0.9

pretrained = True



# -------------- Attribute configuration --------- #
selected_attrs = ["Attractive", "Arched_Eyebrows"]
attr_loss_weight = [1, 0.8]
attr_threshold = [0.6, 0.5]


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
