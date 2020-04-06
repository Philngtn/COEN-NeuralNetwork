import numpy as np 
import DataAT 
import pandas as pd 
import seaborn as sn
import sklearn
import matplotlib.pyplot as plt

class ART1:
    ''' Create class Art1 with 
        n    : Input size (int)
        m    : Internal F2 unit (int) (S)
        Vigi : Vigilance parameter (float) 0 < Vigi <= 1
    '''

    def __init__(self,n=25,m=4,Vigi=0.9):

        # Comparision Field F1(b) layer
        self.F1 = np.zeros(n)
        # Recognition Field F2 layer
        self.F2 = np.zeros(m)
        # Bottom up weights
        self.Wf = np.ones((m,n))*(2/(2-1+n))
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
        # print(I)
        for i in I:
            # Check if nearest memory is above the vigilance level
            
            d = (self.Wb[:,i]*X).sum()/X.sum()
            
            # print(d)
            if d >= self.Vigi:
                # Learn data
                
                self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
                print(self.Wf[:,i])
                self.Wb[:,i] *= X
                
                return self.Wb[:,i], i

        # No match found, increase the number of active units
        # and make the newly active unit to learn data
        if self.active < self.F2.size:
            i = self.active
            self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
            self.Wb[:,i] *= X
            self.active += 1
            return self.Wb[:,i], i

        return None,None


def ConfusionMatrix(predicted):
    actual = np.array(['P1','P2','P3','P1','P4'])
    data = {'y_Actual': actual,
            'y_Predicted': predicted}
    df = pd.DataFrame(data, columns = ['y_Actual','y_Predicted'])
    Confusion_matrix = pd.crosstab(df['y_Actual'],df['y_Predicted'], rownames = ['Actual'], colnames = ['Predicted Class'], margins=True)
    sn.heatmap(Confusion_matrix, annot=True)
    plt.show()
    
    

sample = np.array([DataAT.P1,DataAT.P2,DataAT.P3,DataAT.P1,DataAT.P4])

Vigi = 0.6

Net = ART1(25,3,Vigi)

predicted_results = np.array([])
top_down_template = np.array([])
predicted_class = np.array([])

# noise_percent = 0.17 #add noise from 0.1 to 0.25
    
for i in range(len(sample)):
    # Normal samples
    D, k = Net.learn(sample[i])
  
    predicted_class = np.append(predicted_class,k)
    predicted_results = np.append(predicted_results,'P%d' %(k+1))
    C = np.array(D)
    top_down_template = np.append(top_down_template,C)
    # print_letter(D.reshape(8,8))

predicted_results = np.array(predicted_results)
max_subplot_column = int(np.max(predicted_class))
A = np.split(top_down_template,5)


for i in range(len(A)):
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,5,i+1)
    plt.imshow(A[i].reshape(5,5))
    plt.title("Predicted Class: %i" %(predicted_class[i]+1))
    plt.xticks([])
    plt.yticks([])

plt.show()

ConfusionMatrix(predicted_results)