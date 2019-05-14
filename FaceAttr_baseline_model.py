from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import transforms, models
from GC_resnet import *

"""
Adopt the pretrained resnet model to extract feature of the feature
"""
class FeatureExtraction(nn.Module):
    def __init__(self, pretrained, model_type = "Resnet18"):
        super(FeatureExtraction, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)   
        if model_type == "Resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif model_type == "Resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif model_type == "Resnet101":
            self.model = models.resnet101(pretrained=pretrained)
        elif model_type == "Resnet152":
            self.model = models.resnet152(pretrained=pretrained)
        elif model_type == "GCNet":
            self.model = gc_resnet101(2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, image_batch):
        return self.model(image_batch)



"""
judge the attributes from the result of feature extraction
"""
class FeatureClassfier(nn.Module):
    def __init__(self, selected_attrs, output_dim = 1):
        super(FeatureClassfier, self).__init__()

        self.attrs_num = len(selected_attrs)
        self.selected_attrs = selected_attrs
        output_dim = len(selected_attrs)
        """build full connect layers for every attribute"""
        self.fc_set = {}

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

        self.sigmoid = nn.Sigmoid()

        """
        for attr in selected_attrs:
            self.fc_set[attr] = fc 
            #self.fc_set[attr].to(device)
        """

    def forward(self, x):
        #result_set = []
        x = x.view(x.size(0), -1)  # flatten
        """
        for attr in self.selected_attrs:
            print(self.fc_set[attr])
            res = self.fc_set[attr](x)
            result_set.append(res)
        """
        res = self.fc(x)
        """
        for i in range(len(self.selected_attrs)):
            res[i] = self.sigmoid(res[i])
        """
        res = self.sigmoid(res)
        return res


"""
conbime the extraction and classfier
"""
class FaceAttrModel(nn.Module):
    def __init__(self, model_type, pretrained, selected_attrs):
        super(FaceAttrModel, self).__init__()
        self.featureExtractor = FeatureExtraction(model_type, pretrained)
        self.featureClassfier = FeatureClassfier(selected_attrs)
    
    def forward(self, image):
        features = self.featureExtractor(image)
        results = self.featureClassfier(features)
        return results




