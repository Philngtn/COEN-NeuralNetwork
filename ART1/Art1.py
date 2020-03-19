import numpy as np 
import DataAT 
import pandas as pd 
import seaborn as sn
import sklearn
import matplotlib.pyplot as plt

class ART1:
    ''' Create class Art1 with 
        n    : Input size (int)
        m    : Internal F2 unit (int)
        Vigi : Vigilance parameter (float) 0 < Vigi <= 1
    '''

    def __init__(self,n=8*8,m=19,Vigi=0.5):

        # Comparision Field F1(b) layer
        self.F1 = np.zeros(n)
        # Recognition Field F2 layer
        self.F2 = np.zeros(m)
        # Bottom up weights
        self.Wf = np.ones((m,n))*(1/(1+n))
        # Top-down weights
        self.Wb = np.ones((n,m))
        # Vigilance parameter 
        self.Vigi = Vigi
        # Number of active nodes in F2
        self.active = 0

    def learn(self,X):
    
        # Compute F2
        self.F2 = np.dot(self.Wf,X)
        I = np.argsort(self.F2[:self.active])[::-1]

        for i in I:
            # Check if nearest memory is above the vigilance level
            d = (self.Wb[:,i]*X).sum()/X.sum()
            if d >= self.Vigi:
                # Learn datas
                self.Wb[:,i] *= X
                self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
                return self.Wb[:,i], i

        # No match found, increase the number of active units
        # and make the newly active unit to learn data
        if self.active < self.F2.size:
            i = self.active
            self.Wb[:,i] *= X
            self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
            self.active += 1
            return self.Wb[:,i], i

        return None,None

def noiseAdd(Data,n):
    noise_pixel = int(len(Data)*n)
    for i in range(noise_pixel):
        index = np.random.randint(0,8*8)
        if Data[index] == 0:
            Data[index] = 1
        else:
            Data[index] = 0
    return Data

def ConfusionMatrix(predicted):
    actual = np.array(['A','B','C','D','E',
                       'F','G','H','I','J',
                       'K','L','M','N','O',
                       'P','Q','R','S','T'])

    data = {'y_Actual': actual,
            'y_Predicted': predicted}
    df = pd.DataFrame(data, columns = ['y_Actual','y_Predicted'])
    Confusion_matrix = pd.crosstab(df['y_Actual'],df['y_Predicted'], rownames = ['Actual'], colnames = ['Predicted Class'])
    sn.heatmap(Confusion_matrix, annot=True)
    plt.show()
    
    

def print_letter(Z):
        ''' Print an array as if it was a letter'''
        for row in range(Z.shape[0]):
            for col in range(Z.shape[1]):
                if Z[row,col]:
                    print( '#', end="" )
                else:
                    print( ' ', end="" )
            print( )


sample = np.array([DataAT.A,DataAT.B,DataAT.C,DataAT.D,DataAT.E,
                   DataAT.F,DataAT.G,DataAT.H,DataAT.I,DataAT.J,
                   DataAT.K,DataAT.L,DataAT.M,DataAT.N,DataAT.O,
                   DataAT.P,DataAT.Q,DataAT.R,DataAT.S,DataAT.T])


# for i in range(len(sample)):
#     plt.subplot(4,5,i+1)
#     plt.imshow(sample[i].reshape(8,8))

# plt.subplot(221)
# plt.imshow(sample[6].reshape(8,8))
# plt.title("0% noise")
# plt.subplot(222)
# plt.imshow(noiseAdd(sample[6],0.10).reshape(8,8))
# plt.title("10% noise")
# plt.subplot(223)
# plt.imshow(noiseAdd(sample[6],0.17).reshape(8,8))
# plt.title("17% noise")
# plt.subplot(224)
# plt.imshow(noiseAdd(sample[6],0.25).reshape(8,8))
# plt.title("25% noise")
# plt.show()

Vigi = 0.3

Net = ART1(8*8,20,Vigi)
predicted_results = np.array([])
top_down_template = np.array([])
predicted_class = np.array([])

noise_percent = 0.17 #add noise from 0.1 to 0.25

for i in range(len(sample)):
    # Normal samples
    # D, k = Net.learn(sample[i])
    # Noisy samples
    D, k = Net.learn(noiseAdd(sample[i],noise_percent))
    # print("%c"%(ord('A')+i),"-> class",k)
    predicted_class = np.append(predicted_class,k)
    # predicted_results = np.append(predicted_results,"%c"%(ord('A')+k))
    predicted_results = np.append(predicted_results,k+1)

    C = np.array(D)
    top_down_template = np.append(top_down_template,C)
    # print_letter(D.reshape(8,8))

predicted_results = np.array(predicted_results,dtype= int)

max_subplot_column = int(np.max(predicted_class))
A = np.split(top_down_template,20)
for i in range(len(A)):
    plt.xticks([])
    plt.yticks([])
    plt.subplot(5,4,i+1)
    plt.imshow(A[i].reshape(8,8))
    plt.title("Predicted Class: %i" %(predicted_class[i]+1))
    plt.xticks([])
    plt.yticks([])

plt.show()

ConfusionMatrix(predicted_results)