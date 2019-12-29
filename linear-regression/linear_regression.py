import torch 
import torch.nn as nn
from torch.utils import data 
import torch.nn.functional as F 
from torch.autograd import Variable  
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from torchvision import datasets, transforms
import time
import os

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Breast_Cancer_Classifier(nn.Module):
    def __init__(self):
        super(Breast_Cancer_Classifier, self).__init__()

        self.linear_a = nn.linear(11, 1000)
        self.linear_b = nn.linear(1000, 1)
        

    def forward(self, x):
        x = self.linear_a(x)
        x = F.relu(x)
        y_pred = self.linear(x)
        return y_pred

model = Breast_Cancer_Classifier()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()

for epoch in range(20):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = model(inputs) # Forward pass
    
        # Compute Loss
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        optimizer.step()


class BreastCancerDataset(Dataset):

    def __init__(self):  #Initialise the data, download etc
        df = ...
        self.len(df.shape[0])
        self.x_data = torch.from_numpy(np_array[:,0:-1])
        self.y_data = torch.from_numpy(np_array[:,[-1]])

    def __getitem(self, index):  #return one item on the index
        return self.x_data[index], self.y_data[index]

    def __len__(self): #return the data length
        return self.len

dataset = BreastCancerDataset()
train_loader = DataLoader(dataset=dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            num_workers=2)