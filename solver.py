from __future__ import print_function
from __future__ import division
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import copy
from CelebA import get_loader
import torch.nn.functional as F

from FaceAttr_baseline_model import FaceAttrModel


class Solver(object):
    
    def __init__(self, epoches = 100, batch_size = 64, learning_rate = 0.1,
      model_type = "Resnet18", optim_type = "SGD", momentum = 0.9, pretrained = True,
      selected_attrs=[], image_dir ="./Img", attr_path = "./Anno/list_attr_celeba.txt", log_dir = "./log", 
      use_tensorboard = True, attr_loss_weight = [], attr_threshold = []):

        self.epoches = epoches 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.selected_attrs = selected_attrs
        self.momentum = momentum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.pretrained = pretrained
        self.model_type = model_type
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.build_model(model_type, pretrained)
        self.create_optim(optim_type)
        self.train_loader = None
        self.validate_loader = None
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.attr_loss_weight = torch.tensor(attr_loss_weight)
        self.attr_threshold = attr_threshold
        #self.test_loader = None


    def build_model(self, model_type, pretrained):
        """Here should change the model's structure""" 

        self.model = FaceAttrModel(model_type, pretrained, self.selected_attrs).to(self.device)

        """
        if model_type == "Resnet18":
            self.model = models.resnet18(pretrained = self.pretrained).to(self.device)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, len(self.selected_attrs) * 2)
        elif model_type == "Resnet50":
            self.model = models.resnet50(pretrained= self.pretrained).to(self.device)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2 * len(self.selected_attrs))
        """

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)


    def create_optim(self, optim_type):
        if optim_type == "Adam":
            self.optim = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        elif optim_type == "SGD":
            self.optim = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = self.momentum)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def set_transform(self, mode):
        transform = []
        if mode == 'train':
            transform.append(transforms.RandomHorizontalFlip())
        # the advising transforms way in imagenet
        # the input image should be resized as 224 * 224 for resnet.
        transform.append(transforms.Resize(size=(224, 224)))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform)
        self.transform = transform

    
    # show curve of loss or accuracy
    def show_curve(self, ys, title):
        x = np.array([i for i in range(len(ys))])
        y = np.array(ys)
        plt.figure()
        plt.plot(x, y, c='b')
        plt.axis()
        plt.title('{} curve'.format(title))
        plt.xlabel('epoch')
        plt.ylabel('{} value'.format(title))
        plt.show()

    # self define loss function
    def multi_loss_fn(self, input, target):
        """
        input shape: tensor(batch_size * [attr num...]] 
        target shape: [tensor([... * batch_size]) * attrnum]
        description:
        * input is a output of model [a1, a2 .... ak]
        * target is also a label [b1, b2 ... bk]
        * loss = \sum_i{(ai - bi)**2}
        """
        """
        loss = torch.tensor([0.0])
        for i in range(self.batch_size):
            for j in range(len(self.selected_attrs)):
                loss +=  (input[i][j] - target[j].to(self.device)[i])**2
        """
        #cost_matrix = [1 for i in range(len(self.selected_attrs))]
        total_loss = torch.tensor([0.0])
        for i in range(self.batch_size):
            attr_list = []
            for j, attr in enumerate(self.selected_attrs):
                attr_list.append(target[j][i])
            attr_tensor = torch.tensor(attr_list)
            total_loss += F.binary_cross_entropy_with_logits(input[i],  attr_tensor.type(torch.FloatTensor).to(self.device), weight=self.attr_loss_weight) 
        return total_loss


    def train(self, epoch):
        """
        Return: the average trainging loss value of this epoch
        """
        self.model.train()
        self.set_transform("train")

        # to avoid loading dataset repeatedly
        if self.train_loader == None:
            self.train_loader = get_loader(image_dir = self.image_dir, attr_path = self.attr_path, 
        selected_attrs = self.selected_attrs, mode="train", batch_size=self.batch_size, transform=self.transform)
            print("train_dataset: {}".format(len(self.train_loader.dataset)))

        temp_loss = 0.0
        
        loss_log = {}
        # start to train in 1 epoch
        #print(self.train_loader)
        for batch_idx, samples in enumerate(self.train_loader):
            print("training batch_idx : {}".format(batch_idx))
            images, labels = samples["image"], samples["label"]
            #print(images, labels)
            images= images.to(self.device)
            outputs = self.model(images)
            self.optim.zero_grad()
            
            # get the loss value by the label and output of every attribute
            # sum all the loss values and get a total loss
            # call backward function of the total loss
            loss_dict = {}
            total_loss = None 
            """
            for i, attr in enumerate(self.selected_attrs):
                print(len(outputs[i]), len(labels[i]))
                loss_dict[attr] = self.criterion(outputs[i], labels[i].to(self.device))
                total_loss += loss_dict[attr]
                loss_log[attr] += loss_dict[attr]
            """
            #total_loss = self.criterion(outputs[0], labels)
            total_loss = self.multi_loss_fn(outputs, labels)
            total_loss.backward()

            self.optim.step()
            temp_loss += total_loss.item()
            
        # log the training info 
        """
        for attr in self.selected_attrs:
            loss_log[attr] /= batch_idx + 1
        if self.use_tensorboard:
            for tag, value in loss_log.items():
                self.logger.scalar_summary(tag, value, epoch+1)   
        """
        return temp_loss/(batch_idx + 1)
        
    def evaluate(self):
        """
        Return: correct_dict: save the average predicting accuracy of every attribute 
        """
        
        self.model.eval()
        self.set_transform("validate")
        if self.validate_loader == None:
            self.validate_loader = get_loader(image_dir = self.image_dir, 
                                    attr_path = self.attr_path, 
                                    selected_attrs = self.selected_attrs,
                                    mode="validate", batch_size=self.batch_size, transform=self.transform)
            print("validate_dataset: {}".format(len(self.validate_loader.dataset)))
        
        correct_dict = {}
        for attr in self.selected_attrs:
            correct_dict[attr] = 0
        with torch.no_grad():
            for batch_idx, samples in enumerate(self.validate_loader):
                images, labels = samples["image"], samples["label"]
                images = images.to(self.device)
                outputs = self.model(images)

                # get the accuracys of the current batch
                for i in range(self.batch_size):
                    for j, attr in enumerate(self.selected_attrs):
                        pred = outputs[i].data[j]
                        pred = 1 if pred > self.attr_threshold[j] else 0
                        if pred == labels[j][i]:
                            correct_dict[attr] = correct_dict[attr] + 1
                """  
                for i, attr in enumerate(self.selected_attrs):
                    predicted = outputs[i].data.mmax(1, keepdim = True)[1]
                    correct_dict[attr] += (predicted == labels[i]).sum().detach().numpy()
                """

            # get the average accuracy
            for attr in self.selected_attrs:
                correct_dict[attr] = correct_dict[attr] * 100 / len(self.validate_loader.dataset)
            
        return correct_dict

    def fit(self):
        """
        This function is to combine the train and evaluate, finally getting a best model.
        """
        train_losses = []
        #eval_accs = []
        #eval_losses = []
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        eval_acc_dict = {}
        for attr in self.selected_attrs:
            eval_acc_dict[attr] = []

        for epoch in range(self.epoches):
            running_loss = self.train(epoch)
            print("{}/{} Epoch:  in training process，average loss: {:.4f}".format(epoch + 1, self.epoches, running_loss))
            average_acc_dict = self.evaluate()
            print("{}/{} Epoch: in evaluating process，average accuracy:{}".format(epoch + 1, self.epoches, average_acc_dict))
            train_losses.append(running_loss)
            average_acc = 0.0

            # Record the evaluating accuracy of every attribute at current epoch
            for attr in self.selected_attrs:
                eval_acc_dict[attr].append(average_acc_dict[attr])
                average_acc += average_acc_dict[attr]
            average_acc /= len(self.selected_attrs) # overall accuracy
            
            # find a better model, save it 
            if average_acc > best_acc: 
                best_acc = average_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
    
        # show the curve of trainging loss in every epoch
        #self.show_curve(train_losses, "loss in train")
        #plt.show() 

        # show the curve of evaluating accuracy of every attribute in every epoch
        for attr in self.selected_attrs:
           self.show_curve(eval_acc_dict[attr], "accuracy in evaluate")

        # load best model weights
        model_save_path = "./" + self.model_type + "-best_model.pt"
        self.model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, model_save_path)
        

    def predict(self, image):
        self.model.load_state_dict(torch.load("./" + self.model_type + "-best_model.py"))
        self.model.eval()
        with torch.no_grad():
            self.set_transform("predict")
            output = self.model(self.transform(image))
            pred = output.data.max(1, keepdim = True)[1]
            return pred
