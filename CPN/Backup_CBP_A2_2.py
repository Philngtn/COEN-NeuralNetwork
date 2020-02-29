from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt
from Data import X,y
from scipy import sparse

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z
## One-hot coding
def convert_labels(y, C):
    Y = sparse.coo_matrix((np.ones_like(y),(y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y
# cost or loss function
def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

def calculate_Z(U,V,x,y):
    return np.dot(U.T,x) + np.dot(V.T,y)

def winning_Z(Z):
    return np.argmax(Z,axis=0)

def saturate_input(X):
    X_square = X**2
    A = np.sum(X_square,axis = 0)
    return np.divide(X,np.sqrt(A))

#Seperate Training Data and Test Data
Total_sample = X.shape[1]

Seperator = int(round(Total_sample*0.7))

Train_samples = X[:,0:Seperator]
Test_samples = X[:,Seperator:Total_sample]

Train_labels = y[0:Seperator]
Test_labels = y[Seperator:Total_sample]

#Initial Condition
d1 = d4 = 2  # x and x' input layers
d3 = 700 # size of Kohonen layer 3
d5 = d2 = C = 2 # y and y' output layer 

# initialize parameters randomly
U_13     = 0.1*np.random.rand(d1, d3) #2xZ_layers
V_53     = 0.1*np.random.rand(d5, d3) #2xZ_layers
W_32     = 0.1*np.random.rand(d3, d2) #Z_layers x2
W_hat_34 = 0.1*np.random.rand(d3, d4) #Z_layers x2

#Calculate Accuracy and Loss Before Training
Y_test = convert_labels(Test_labels, C)
Z = calculate_Z(U_13,V_53,Test_samples,np.zeros_like(Y_test))
Y_dash = np.dot(np.array(W_32).T,Z)
Yhat = softmax(Y_dash) 
predicted_class = np.argmax(Yhat, axis=0)
acc = (100*np.mean(predicted_class == Test_labels))
print('Pre-training accuracy: %.2f %%' %acc)


#Training Section
Y_Train = convert_labels(Train_labels, C) #one hot coding 
Train_samples = saturate_input(Train_samples)
Y_Train = saturate_input(Y_Train)

Y_dash = np.zeros_like(Y_Train)
X_dash = np.zeros_like(Train_samples)

alpha_X_dash = w_X_dash = 0.5
beta_Y_dash = 0.2
w_Y_dash = 0.27
 
epoch = 2
for t in range(epoch):
    # Feedforward
    for i in range(Train_samples.shape[1]):
        Z = calculate_Z(U_13,V_53,Train_samples[:,i],Y_Train[:,i])
        if i > Train_samples.shape[1]*0.2:
            win_node = winning_Z(Z)
            
            U_13[:,win_node] = U_13[:,win_node] + alpha_X_dash*(Train_samples[:,i] - U_13[:,win_node]) 
            V_53[:,win_node] = V_53[:,win_node] + beta_Y_dash*(Y_Train[:,i] - V_53[:,win_node]) 
        else:
            for j in range(U_13.shape[1]):
                U_13[:,j]    = U_13[:,j]    + alpha_X_dash*(Train_samples[:,i] - U_13[:,j])
                V_53[:,j]    = V_53[:,j]    + beta_Y_dash*(Y_Train[:,i] - V_53[:,j])  
            
          
    for i in range(Train_samples.shape[1]):
        Z = calculate_Z(U_13,V_53,Train_samples[:,i],Y_Train[:,i])
        if i > Train_samples.shape[1]*0.2:
            win_node = winning_Z(Z)

            W_32[win_node,:]     = W_32[win_node,:]     + w_Y_dash*(Y_Train[:,i] - W_32[win_node,:]) 
            W_hat_34[win_node,:] = W_hat_34[win_node,:] + w_X_dash*(Train_samples[:,i] - W_hat_34[win_node,:]) 

            X_dash[:,i] = np.dot(np.array(W_hat_34[win_node,:]).T,win_node) + Train_samples[:,i]/np.sum(Train_samples[:,i])
            Y_dash[:,i] = np.dot(np.array(W_32[win_node,:]).T,win_node)  +    Y_Train[:,i]/np.sum(Y_Train[:,i])

       
        else:
            for k in range(W_32.shape[1]):
                W_32[k,:]     = W_32[k,:]    +  w_Y_dash*(Y_Train[:,i]- W_32[k,:]) 
                W_hat_34[k,:] = W_hat_34[k,:] + w_X_dash*(Train_samples[:,i] - W_hat_34[k,:]) 
            
            X_dash[:,i] = np.dot(np.array(W_hat_34).T,Z) + Train_samples[:,i]/np.sum(Train_samples[:,i])
            Y_dash[:,i] = np.dot(np.array(W_32).T,Z)  + Y_Train[:,i]/np.sum(Y_Train[:,i])
            

#Calculate Accuracy and Loss After Training
Y_test = convert_labels(Test_labels, C)
Z = calculate_Z(U_13,V_53,Test_samples,np.zeros_like(Y_test))
Y_dash = np.dot(np.array(W_32).T,Z) 
Yhat = softmax(Y_dash) 
predicted_class = np.argmax(Yhat, axis=0)
acc = (100*np.mean(predicted_class == Test_labels))
print('Post-training accuracy: %.2f %%' %acc)


# Visualize 
xm = np.arange(-10,10, 0.025)
xlen = len(xm)
ym = np.arange(-10, 10, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)

xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

X0 = np.vstack((xx1, yy1))
#-----------------------------

Z = calculate_Z(U_13,V_53,X0,np.zeros_like(X0))
Z2 = np.dot(np.array(W_32).T,Z) 

# predicted class 
Z = np.argmax(Z2, axis=0)
Z = Z.reshape(xx.shape)

CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

for i in range(Test_labels.shape[0]):
    if Test_labels[i] == 1:
        plt.plot(Test_samples[0,i],Test_samples[1,i],'.r',markersize = 1)
    else:
        plt.plot(Test_samples[0,i],Test_samples[1,i],'.b',markersize = 1)

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
plt.title('#Kohonen layers = %d, accuracy = %.2f %%' %(d3, acc))
# plt.axis('equal')
# display(X[1:, :], original_label)
# fn = 'ex_res'+ str(d3) + '.png'
# plt.savefig(fn, bbox_inches='tight', dpi = 600)
plt.show()