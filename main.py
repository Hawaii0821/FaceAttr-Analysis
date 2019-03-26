from solver import Solver

# args setting
epoches = 3
batch_size = 64
learning_rate = 0.5
model_type = "Resnet18"
optim_type = "SGD"
momentum = 0.9
pretrained = True
selected_attrs = ["Attractive"]
# You should download the celeba dataset in the root dir.
image_dir = "./Img/img_align_celeba/" 
attr_path = "./Anno/list_attr_celeba.txt"

if __name__ == "__main__":
    solver = Solver(epoches=epoches, batch_size=batch_size, learning_rate=learning_rate,
                    model_type=model_type, optim_type=optim_type, momentum=momentum, pretrained=pretrained,
                    selected_attrs=selected_attrs, image_dir = image_dir, attr_path=attr_path)
    solver.fit()
