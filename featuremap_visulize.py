import matplotlib.pyplot as plt
plt.switch_backend("agg")
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import sys
import numpy as np
from PIL import Image
from FaceAttr_baseline_model import FaceAttrModel
import config as cfg 
import cv2 
import torch.nn as nn

class FeatureExtractor():
    """ 
    Class for extracting activations and
        registering gradients from targetted intermediate layers 
    """

    def __init__(self, model, target_layers):
        self.model = model._modules["model"]
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        print(self.model._modules.items())

        for name, module in self.model._modules.items():
            x = module(x)
            # print(name)
            # print(module)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ 
    Class for making a forward pass, and getting:
        1. The network output.
        2. Activations from intermeddiate targetted layers.
        3. Gradients from intermeddiate targetted layers. 
    """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.featureExtractor, target_layers)

    def get_gradients(self):
        # print("get gradient {}".format(self.feature_extractor.gradients))
        return self.feature_extractor.gradients  # TODO

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        print(output.size())
        output = output.view(output.size(0), -1)
        output = self.model.featureClassfier(output)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask, save_path="./cam_based/grad_cam.jpg"):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))
    print("image has saved at {}".format(save_path))
    plt.imshow(Image.open(save_path))
    plt.axis('off')
    # plt.show()


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        print(features)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.featureExtractor.zero_grad()
        self.model.featureClassfier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # print(cam.shape)
        # print(target.shape)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # print(cam)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        print(np.max(cam))
        cam = cam / np.max(cam)
        return cam

def visulize(model, target_layer_names, image_path, save_path, target_index):
    """
    input:
        - model: the visulizing model
        - target_layer_names: the visulizing layer name
        - image_path: the test image path
        - save_path: save the visulizing result
        - target_index: the index of target attribute
    """
    assert isinstance(target_layer_names, list)
    grad_cam = GradCam(model=model, target_layer_names=target_layer_names, use_cuda=True)
    img = cv2.imread(image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input_image = preprocess_image(img)
    mask = grad_cam(input_image, target_index)
    # print("mask :{}".format(mask))
    show_cam_on_image(img, mask, save_path)

if __name__ == '__main__':
    # the test image path
    image_path = "001735.jpg"

    # the output image path
    save_path = "001735-35.jpg"

    # the index of the target attributes. 35 denotes hearing_hat
    target_index = 35
    
    # define the model
    model = FaceAttrModel("Resnet18", pretrained=False, selected_attrs=cfg.selected_attrs)
    
    # load the model state dict, you should prepare the pretrained model.
    state_dict = torch.load("./result/resnet18.pth")
    model.load_state_dict(state_dict)

    # visulize the 7'th part of the network. Why 7 ? You can print the state dict to watch more.
    visulize(model, ["7"], image_path, save_path, target_index)