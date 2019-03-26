from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import transforms, models

"""
Adopt the pretrained resnet model to extract feature of the feature
"""
class FeatureExtraction(nn.Module):
    def __init__(self, model_type, pretrained):
        super(FeatureExtraction, self).__init__()
        if model_type == "Resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_type == "Resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        
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

        """build full connect layers for every attribute"""
        self.fc_set = {}

        fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

        for attr in selected_attrs:
            self.fc_set[attr] = fc 
            #self.fc_set[attr].to(device)
    
    def forward(self, x):
        result_set = {}
        x = x.view(x.size(0), -1)  # flatten
        for attr in self.selected_attrs:
            res = self.fc_set[attr](x)
            result_set[attr] = res
        return result_set


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




