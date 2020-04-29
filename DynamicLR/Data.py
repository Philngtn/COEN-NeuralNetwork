import numpy as np 
from scipy import sparse

labeled_class = np.array([[1,1,0],[1,2,0],[2,-1,1],[2,0,1],[-1,2,2],
                          [-2,1,2],[-1,-1,3],[-2,-2,3],[-1,-3,4],[3,3,4]])

Data = np.array([-2,-2,3])
# Samples
N = 1000
for i in range(N-1):
    a = np.random.randint(0,10)
    temp = np.array(labeled_class[a])
    Data = np.vstack((Data,temp))
    
X = Data[:,0:2].T
y = Data[:,2]




