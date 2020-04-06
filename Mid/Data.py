import numpy as np 
from scipy import sparse

labeled_class = np.array([[1,1,0],[1,2,0],[2,-1,1],[2,0,1],[-1,2,2],[-2,1,2],[-1,-1,3],[-2,-2,3]])
# print(labeled_class[0])
Data = np.array([-2,-2,3])
# Samples
N = 1000
for i in range(N-1):
    a = np.random.randint(0,8)
    temp = np.array(labeled_class[a])
    Data = np.vstack((Data,temp))
    
# print(Data)
X = Data[:,0:2].T
y = Data[:,2]
# print(X)
# print(y)

# def convert_labels(y, C):
#     Y = sparse.coo_matrix((np.ones_like(y),
#         (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
#     return Y

# print(convert_labels(y,5))



