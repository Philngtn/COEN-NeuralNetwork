import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np 


from data_train import Data, Output

################## Prepare Data #################################

x_train = torch.FloatTensor(Data[0:3640])   #70% data set 3640
y_train = torch.FloatTensor(Output[0:3640])

x_test =  torch.FloatTensor(Data[3640:5200])  #15%
y_test =  torch.FloatTensor(Output[3640:5200])

# x_train = torch.FloatTensor(Data[0:40])   #70% data set 3640
# y_train = torch.FloatTensor(Output[0:40])

# x_test = torch.FloatTensor(Data[40:60])  #15%
# y_test =  torch.FloatTensor(Output[40:60])

################## Neural Network #################################

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Feedforward, self).__init__()

            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output


NN = Feedforward(2,10,1)
criterion = torch.nn.BCELoss()

for lr in np.arange(0, 1, 0.05):
    optimizer = torch.optim.SGD(NN.parameters(),lr=0.1)
    # optimizer = torch.optim.Adam(NN.parameters(),lr=0.9, betas=(0.9, 0.999), eps=1e-08)
    loss_value=[]
    ep = []

    NN.eval()
    y_pred = NN(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training' , before_train.item())

    NN.train()
    epoch = 100
    for epoch in range(epoch):

        optimizer.zero_grad()
        # Forward pass
        y_pred = NN(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        loss_value.append(loss)
        ep.append(epoch)
        # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()


    NN.eval()
    y_pred = NN(x_test)
    after_train = criterion(y_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item())

plt.plot(ep,loss_value)
plt.show()

