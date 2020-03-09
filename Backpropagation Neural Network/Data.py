#READ collum + row in excel to n-D list in 2 ways
import xlrd
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

workbook = xlrd.open_workbook("Data.xlsx")
sheet = workbook.sheet_by_name("Sheet1")

rowcount = sheet.nrows
colcount = sheet.ncols

# print(rowcount)
# print(colcount)

Data =[]
Output = []
Data_Compress = []

for curr_col in range(0, colcount, 1): #5300 columns
    col_data = []
    col_data_out =[]

    for curr_row in range(0, rowcount, 1): #3 rows
        if curr_row != 2:
            data = sheet.cell_value(curr_row, curr_col) #take data from 1 single cell
            col_data.append(data)
        else:
            data_out = sheet.cell_value(curr_row, curr_col) #take data from 1 single cell
            col_data_out.append(data_out)
            
    Data.append(col_data)
    Data.append(col_data_out)


# print(Data)

for i in range(0,len(Data),2):
    data_select = []
    for j in range(2):
        data_select.append(Data[i+j])
    Data_Compress.append(data_select)

    
# print(Data_Compress)
# print(len(Data))

# Data will store like this Data = [[[0.737784474287975, 0.288032444846822], [1.0]],...,[[1.24002004818493, 0.0990697069733273], [1.0]]]

N = int(len(Data_Compress)/2) # number of points per class
d0 = 2 # dimensionality
C = 2 # number of classes

X = np.empty((d0,N*C)) # data matrix (each row = single example)
y = np.empty(N*C, dtype='uint8') # Label class


for i in range(len(Data_Compress)):
    X[:,i] = np.c_[Data_Compress[i][0][0], Data_Compress[i][0][1]]
    y[i] =   Data_Compress[i][1][0]

# print(range(y.shape[0]))
# N = X.shape[1];
# M = int(round(N*0.3))
# print(M)

# TrainData = X[:,1:3640];
# print(TrainData.shape)



# for i in range(y.shape[0]):
#     print(i)

# for i in range(C_Data.shape[1]):
#     if C_Data[2][i] == 1:
#         plt.plot(C_Data[0,i],C_Data[1,i],'.r',markersize = 1)
#     else:
#         plt.plot(C_Data[0,i],C_Data[1,i],'.b',markersize = 1)


# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# cur_axes = plt.gca()
# cur_axes.axes.get_xaxis().set_ticks([])
# cur_axes.axes.get_yaxis().set_ticks([])
# plt.show()