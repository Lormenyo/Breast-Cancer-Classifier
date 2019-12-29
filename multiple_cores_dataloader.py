# Generate your data on multiple cores in real time and feed it right away to your deep learning model.

#Dictionary called partition 
#partition['train'] = list of training IDs 
#partition['test'] = list of testing IDs 

#Dictionary called labels
#labels[ID] = label of the given id'

import torch
from torch.utils import data

class dataset(data.Dataset):
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        #labels and the list of IDs that we wish to generate at each pass.

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data using the index'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Datasets
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}# IDs
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}# Labels

# Generators
training_set = dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        [...]

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]