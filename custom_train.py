#Required imports
import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from  darknet import Darknet
from util import *

#Setting up the training parameters

config_file = 'cfg/yolov3.cfg'                 #Configuration file path
pretrained_weights = 'yolov3.weights'          #Path to pretrained weights
dataset_path = ''                              #Path to custom dataset
annotations_path = ''                          #Path to annotations
class_names_path = ''                          #Path to class names file
num_classes = count_classes(class_names_path)  #Setting the number of classes in our custom dataset
inp_dim = 416                                  #Setting desired input image size
batch_size = 8                                 #Setting the batch size
num_epochs = 400                               #Setting the number of training epochs
learning_rate = 0.001                          #Setting the learning rate

#Loading the YOLOv3 model from the configuration file
model = Darknet(config_file)

#Load the pre-trained weights
model.load_weights(pretrained_weights)

#Load the class names
class_names = load_classes(class_names_path)

#Setting up the optimizer and loss functions
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
criterion = nn.MSELoss()

'''
Create the dataloader for the custom datset. We will need to implement your own custom dataset class.
The dataset class should handle loading images and annotations and apply necessary transformations to the data
'''
custom_dataset = CustomDataset(dataset_path, annotations_path)
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle = True, num_workers=4)

#Setting the model to training mode
model.train()

for epoch in range(num_epochs):
    #Initializing the total loss for the epoch
    total_loss = 0

    #Iterate ove the training dataset
    for batch_idx, (images, targets) in enumerate(data_loader):
        #Transfer the data to the GPU if available
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        #Zero the gradients
        optimizer.zero_grad()

        #Forward pass
        outputs = model(images)

        #Calculate the loss
        loss = criterion(outputs, targets)

        #Backward pass
        loss.backward()

        #Update the weights
        optimizer.step()

        #Accumulate the loss
        total_loss += loss.item()

        #Print the training progress
        print('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(data_loader), loss.item()))

        #Print the average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        print('Epoch [{}/{}], Average Loss: {:.4f}'.format(epoch+1, num_epochs, avg_loss))

        #Save the custom weights
        torch.save(model.state_dict(), 'custom_weights.pth')