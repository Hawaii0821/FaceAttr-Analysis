import torch
import torchvision
#from torchvision import transforms, utils
import pandas as pd
#from skimage import io, transform
from torch.utils import data
import os
import random
from PIL import Image


# 根据Eval文件里头的数据划分,表示该数据集从上到下划分的终点
train_end_index = 256 + 1
validate_end_index = 320 + 1
test_end_index = 384 + 1

class CelebA(data.Dataset):
    
    # each image is  218*178;

    def __init__(self, attr_file, selected_attrs, image_folder,
                transform, mode = "train"):

        #self.read_bbox_file(bbox_file)
        self.attr_file = attr_file
        self.image_folder = image_folder
        self.transform = transform
        self.selected_attrs = selected_attrs

        self.train_dataset = []
        self.validate_dataset = []
        self.test_dataset = []

        self.idx2attr = {}
        self.attr2idx = {}

        self.preprocess()
        self.mode = mode
        if mode == "train":
            self.num_images = len(self.train_dataset)
        elif mode == "validate":
            self.num_images = len(self.validate_dataset)
        elif mode == "test":
            self.num_images = len(self.test_dataset)
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, index):
        """Return image data (tensor) and labels (dict)"""
        dataset = self.train_dataset
        if self.mode == 'validate':
            dataset = self.validate_dataset
        elif self.mode == "test":
            dataset = self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_folder, filename)) 
        return {"image": self.transform(image), "label": label}

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_file, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1024)
        random.shuffle(lines)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = {}
            # 将结果按属性进行保存，作为一个字典标签
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                val = int(values[idx])
                if val == -1:
                    val = 0
                label[attr_name] = torch.LongTensor(val)

            # 根据行索引的位置来划分训练集，验证集，测试集
            if (i+1) < train_end_index:
                self.train_dataset.append([filename, label])
            elif (i + 1) < validate_end_index:
                self.validate_dataset.append([filename, label])
            elif (i + 1) < test_end_index:
                self.test_dataset.append([filename, label])
            elif i >= test_end_index:
                break 
        print('Finished preprocessing the CelebA data set...')


    # ---------------------------------------------------------------#
    # 以下函数写完发现暂时用不到。。。。可能后续可以自己实现一个人脸检测吧 #
    # ---------------------------------------------------------------#
    def read_bbox_file(self, bbox_file):
        with open(bbox_file, 'r') as f:
            bbox_attr = f.readlines()
            self.bbox_nums = bbox_attr[0]
            bbox_cols = bbox_attr[1].split()
            bbox_attr = bbox_attr[2:]
            frame = []
            for attr in bbox_attr:
                row_sample = attr.split()
                frame.append(row_sample)
            bbox_frame = pd.DataFrame(frame, columns=bbox_cols)
            self.bbox_frame = bbox_frame
            return bbox_frame
    
    def read_attr_file(self, attr_file):
        with open(attr_file, 'r') as f:
            attrs = f.readlines()
            self.image_num = attrs[0]
            attr_cols = ["ImageID"]
            attr_cols = attr_cols + attrs[1]
            attrs = attrs[2:]
            frame = []
            for attr in attrs:
                row_sample = attr.split()
                frame.append(row_sample)
            face_atrr_frame = pd.DataFrame(frame, columns = attr_cols)
            return face_atrr_frame

    def read_landmarks_file(self, landmarks_file):
        with open(landmarks_file, 'r') as f:
            landmarks = f.readlines()
            self.image_num = landmarks[0]
            attr_cols = ["ImageID"]
            attr_cols = attr_cols + landmarks[1]
            landmarks = landmarks[2:]
            frame = []
            for landmark in landmarks:
                row_sample = landmark.split()
                frame.append(row_sample)
            landmark_frame = pd.DataFrame(frame, columns = attr_cols)
            return landmark_frame

    def read_partition_file(self, partition_file):
        with open(partition_file, 'r') as f:
            info = f.readlines()
            frame = []
            for line in info:
                frame.append(line.split())
            partition_frame = pd.DataFrame(frame, columns = ["ImageID", "Type"])
            return partition_frame
     

# 218 * 178
def get_loader(image_dir, attr_path, selected_attrs,
               batch_size, mode='train', num_workers=1, transform = None):
    """Build and return a data loader."""
    dataset = CelebA(attr_path, selected_attrs, image_dir,transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader