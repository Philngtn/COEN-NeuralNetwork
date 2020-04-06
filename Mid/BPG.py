from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt
from Data import X,y
from scipy import sparse
import pandas as pd 
import seaborn as sn
import sklearn

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

#Seperate Training Data and Test Data
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
d2 = C = 4
# initialize parameters randomly
W1 = 0.01*np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))
#---------------------------------------------
#Testing Section
Y_test = convert_labels(Test_labels, C)
#Calculate Accuracy and Loss Before Training
Z1 = np.dot(W1.T, Test_samples) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
Yhat = softmax(Z2)
predicted_class = np.argmax(Yhat, axis=0)
acc = (100*np.mean(predicted_class == Test_labels))
print('Pre-training accuracy: %.2f %%' %acc)
loss = cost(Y_test, Yhat)
print("Loss Pre-Training: %f" %(loss))

#---------------------------------------------
#Training Section
Y_Train = convert_labels(Train_labels, C)
N = Train_samples.shape[1]
eta = 0.3 # learning rate
loss_capture = []
count = []
for i in range(1000):
    ## Feedforward
    Z1 = np.dot(W1.T, Train_samples) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    Yhat = softmax(Z2)

    # print loss after each 1000 iterations
    if i %100 == 0:
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

    # Gradient Descent update
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2

#--------------------------------------------
#Testing Accuracy
#Testing Section
Y_test = convert_labels(Test_labels, C)
#Calculate Accuracy and Loss After Training
Z1 = np.dot(W1.T, Test_samples) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
Yhat = softmax(Z2)
predicted_class = np.argmax(Yhat, axis=0)
# print(predicted_class)
# print(Test_labels)
acc = (100*np.mean(predicted_class == Test_labels))
print('Post-training accuracy: %.2f %%' %acc)
loss = cost(Y_test, Yhat)
print("Loss Post-Training: %f" %(loss))

def ConfusionMatrix(actual,predicted):
    data = {'y_Actual': actual,
            'y_Predicted': predicted}
    df = pd.DataFrame(data, columns = ['y_Actual','y_Predicted'])
    Confusion_matrix = pd.crosstab(df['y_Actual'],df['y_Predicted'], rownames = ['Actual'], colnames = ['Predicted Class'],margins=True)
    sn.heatmap(Confusion_matrix, annot=True)
    plt.show()

ConfusionMatrix(Test_labels,predicted_class)


plt.plot(count,loss_capture)
plt.title('Loss function after each 100 epoch')
plt.show()


# Visualize 
xm = np.arange(-5, 5, 0.025)
xlen = len(xm)
ym = np.arange(-5, 5, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)

xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

X0 = np.vstack((xx1, yy1))

Z1 = np.dot(W1.T, X0) + b1 
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2

# predicted class 
Z = np.argmax(Z2, axis=0)
Z = Z.reshape(xx.shape)

CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

# Plot also the training points
# plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# X = X.T



for i in range(Test_labels.shape[0]):
    if Test_labels[i] == 0:
        plt.plot(Test_samples[0,i],Test_samples[1,i],'.g',markersize = 10)
    elif Test_labels[i] == 1:
        plt.plot(Test_samples[0,i],Test_samples[1,i],'.b',markersize = 10)
    elif Test_labels[i] == 2:
        plt.plot(Test_samples[0,i],Test_samples[1,i],'.y',markersize = 10) 
    elif Test_labels[i] == 3:
        plt.plot(Test_samples[0,i],Test_samples[1,i],'.r',markersize = 10)

# plt.axis('off')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xticks(())
plt.yticks(())
plt.title('#Hidden units = %d, Accuracy = %.2f %%' %(d1, acc))
# plt.axis('equal')
# display(X[1:, :], original_label)
fn = 'ex_res'+ str(d1) + '.png'
# plt.savefig(fn, bbox_inches='tight', dpi = 600)
plt.show()