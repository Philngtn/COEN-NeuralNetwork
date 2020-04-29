from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt
from Data import X,y
from scipy import sparse
import pandas as pd 
import seaborn as sn
import sklearn

# Functions

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z
## One-hot coding
def convert_labels(y, C):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y
# cost or loss function
def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

def testing():
    Y_test = convert_labels(Test_labels, C)
    #Calculate Accuracy and Loss Before Training
    Z1 = np.dot(W1.T, Test_samples) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    Yhat = softmax(Z2)
    predicted_class = np.argmax(Yhat, axis=0)
    acc = (100*np.mean(predicted_class == Test_labels))
    loss = cost(Y_test, Yhat)
    return acc, loss, predicted_class

#--------------------------------------------
#Seperating Training Data and Test Data
Total_sample = X.shape[1]

Train_range = int(round(Total_sample*0.7))

Train_samples = X[:,0:Train_range]
Train_labels = y[0:Train_range]
#---------------------------------------------
Test_samples = X[:,Train_range:Total_sample]
Test_labels = y[Train_range:Total_sample]

#---------------------------------------------
#Initial Condition
d0 = 2
d1 = h = 10 # size of hidden layer
d2 = C = 5
# initialize parameters randomly
W1 = 0.01*np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))

#---------------------------------------------
#Testing Section
acc, loss, predicted_class = testing()
print('Pre-training accuracy: %.2f %%' %acc)
print("Loss Pre-Training: %f" %(loss))

#---------------------------------------------
#Training Section
Y_Train = convert_labels(Train_labels, C) #one-hot coding 
N = Train_samples.shape[1]
loss_capture = []

F_x = []
F_rec = 10e9999
F_lec = []
count = []
epoch = 200

# Initial dynamic learning rate
B = 2*np.random.rand() #0<B<2 
delta = 2
eta = 2 # learning rate
G = []
for i in range(epoch):
    ## Feedforward
    Z1 = np.dot(W1.T, Train_samples) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    Yhat = softmax(Z2)

    # print loss after each 1000 iterations
    if i %1 == 0:
        # compute the loss: average cross-entropy loss
        loss_capture.append(cost(Y_Train, Yhat))
        count.append(i)
        # print("iter %d, loss: %f" %(i, loss))

    # backpropagation
    E2 = (Yhat - Y_Train)/N
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis = 1, keepdims = True)
    
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0 # gradient of ReLU
    dW1 = np.dot(Train_samples, E1.T)
    db1 = np.sum(E1, axis = 1, keepdims = True)

    # F_x = cost(Y_Train,Yhat)
    # G.append(F_x)
    # #Step 1 Function Evaluation
    # if F_x < F_rec:
    #     F_rec = F_x
    # else: 
    #     F_rec = F_rec
    # #Step 2 Sufficient Descent
    # if F_x < (F_rec - delta/2):
    #     delta = delta
    # #Step 3 Oscillation detection 
    # if eta>B:
    #     delta = delta/2
    # #Step 4 Iterate update
    # F_lev = F_rec - delta
    # eta = B*(F_x - F_lev)/np.max(G)

    # Gradient Descent update
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2

#--------------------------------------------
#Testing Accuracy
acc, loss, predicted_class = testing()
print('Post-training accuracy: %.2f %%' %acc)
print("Loss Post-Training: %f" %(loss))
#--------------------------------------------
#Ploting loss function

plt.plot(count,loss_capture) 
plt.title('Loss function after 200 epoch, Inital dynamic lr=2')
plt.show()
