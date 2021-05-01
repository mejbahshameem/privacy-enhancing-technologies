import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, ConcatDataset
from torch.utils.data import TensorDataset, DataLoader
import os

#Dataset class for feature vector of MLP(ATTACK model)
class Dataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, features, labels):
            'Initialization'
            self.features = features
            self.labels = labels
    
      def __len__(self):
            'Denotes the total number of samples'
            return len(self.labels)
    
      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
    
            # Load data and get label
            X = self.features[index]
            y = self.labels[index]
    
            return X, y
        
        
def process_image_dataset(trainset, testset, image_batch_size):
    
    datasets=[]
    datasets.append(trainset)
    datasets.append(testset)
    # Conact all data samples together different train and test set for image classification is not needed 
    fulldataset = ConcatDataset(datasets)
    
    #All the names of dataset after split is maintained exactly with Ahmed's paper
  
    Dshadow, Dtarget = random_split(fulldataset, (int(len(fulldataset)/2), int(len(fulldataset)/2)))
    Dtrainshadow, Doutshadow = random_split(Dshadow, (int(len(fulldataset)/4), int(len(fulldataset)/4)))
    Dtrain, Dnonmember = random_split(Dtarget, (int(len(fulldataset)/4), int(len(fulldataset)/4)))
    Dtrainloader = torch.utils.data.DataLoader(Dtrain, image_batch_size,
                                              shuffle=True, num_workers=0)
    
    Dnonmemberloader = torch.utils.data.DataLoader(Dnonmember, image_batch_size,
                                              shuffle=True, num_workers=0)
    
    Dtrainshadowloader = torch.utils.data.DataLoader(Dtrainshadow, image_batch_size,
                                              shuffle=True, num_workers=0)
    
    Doutshadowloader = torch.utils.data.DataLoader(Doutshadow, image_batch_size,
                                              shuffle=True, num_workers=0) 
    
    Dshadowloader = torch.utils.data.DataLoader(Dshadow, image_batch_size,
                                              shuffle=True, num_workers=0) 
    
    Dtargetloader = torch.utils.data.DataLoader(Dtarget, image_batch_size,
                                              shuffle=True, num_workers=0) 
    
    return Dtrainloader, Dnonmemberloader,  Dtrainshadowloader, Doutshadowloader


def process_posterior_dataset(model, Dmemberloader, Dnonmemberloader, posterior_batch_size, image_batch_size, shuffle):
    
    #mlp_X will be prepared for feeding in attack model that contains three top posteriors
    #mlp_Y will contain the label 1-member/0-nonmember
    #Dataset will be Shuffled (mandatory) for shadow posteriors but for making target posteriors it will be set to false as, for target we need just the prediction we won't train attack model on target data
    mlp_X = torch.empty(0,3).to(torch.float32)
    mlp_one = torch.ones(len(Dmemberloader)*image_batch_size,1, dtype=int).to(torch.float32)
    mlp_zero = torch.zeros(len(Dnonmemberloader)*image_batch_size,1, dtype=int).to(torch.float32)
    mlp_Y = torch.cat((mlp_one,mlp_zero),0)
   
    #Get all the posteriors from already trained model, rank them high to low and make feature vector mlp_X 
    with torch.no_grad():
            for data in Dmemberloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(image_batch_size):
                    values, _ = torch.sort(outputs[i])
                    mlp_X = torch.cat((torch.flip(values[-3:],[0]).view(1,3), mlp_X))
                    
            for data in Dnonmemberloader:
              
                images, labels = data
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(image_batch_size):
                    values, _ = torch.sort(outputs[i])
                    mlp_X = torch.cat((torch.flip(values[-3:],[0]).view(1,3), mlp_X))
           
    mlp_X = torch.flip(mlp_X,[0])
    
    mlp_dataset = Dataset(mlp_X, mlp_Y)

    Dposteriorloader = torch.utils.data.DataLoader(mlp_dataset, posterior_batch_size,
                                      shuffle=shuffle)
        
    return Dposteriorloader 