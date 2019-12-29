import torch 
import torch.nn as nn
from torch.utils import data 
import torch.nn.functional as F 
from torch.autograd import Variable as V 
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from torchvision import datasets, transforms
import time
import os

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        #declare your layer as class variables here
        # For example :
        # self.linear = nn.linear(input_dim, output_dim)

    def forward(self, x):
        pass
        # This has the training steps and activation functions for the model Like below:
        #x = self.linear(x)
        #x = F.relu(x)
        #x = self.linear2(x)
        #return x

model = Network()

###################### EXAMPLE MNIST CLASSIFIER ###############################
class MNIST_classifier(nn.Module):
    def __init__(self):
        super(MNIST_classifier, self).__init__()

        #layers of the neural network
        self.input_layer = nn.Linear(784, 1000)
        self.output_layer = nn.Linear(1000, 10)

    def forward(self, x):
        x = x.view(-1, 784) 
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        output = F.softmax(x, dim=1)
        return output


      # The backward() function is then defined for you autmatically via autograd

model = MNIST_classifier() 

batch_size = 64

training_set = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)


############# Training ############################
model.train() #set the model to train mode

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


start_time = time.time()
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_set, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward + backward + optimize
        outputs = model(inputs)   #is the same is model.forwward

        # computing the loss
        loss = criterion(outputs, labels)
        # back propagation 
        loss.backward()
        optimiser.step() #update parameters based on loss

        #  reset the gradients to zero before moving forward, because PyTorch accumulates gradients.
        optimiser.zero_grad()

        # print statistics
        running_loss += loss.item()
        if i % 64 == 0:    # print every 64 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

end_time = time.time()
print('Finished Training')
print('Training Time: ', end_time - start_time)


######################### TESTING ##########################################
test_batch_size = 1000
test_set = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test(model=model, device=device, test_loader=test_set)

summary(model, (784, 1))  #estimated size of the model
torch.save(model.state_dict(), "two_layer_model_b.path")
size_of_model = os.path.getsize("two_layer_model_b.path") /1000000
print("size of model: ", size_of_model)