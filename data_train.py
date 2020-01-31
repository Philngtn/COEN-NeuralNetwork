#READ collum + row in excel to n-D list in 2 ways
import xlrd
import torch


workbook = xlrd.open_workbook("Data.xlsx")
sheet = workbook.sheet_by_name("Sheet1")

rowcount = sheet.nrows
colcount = sheet.ncols

# print(rowcount)
# print(colcount)

Data =[]
Output = []

# for curr_row in range(0, rowcount, 1): #3 rows
#     row_data = []

#     if curr_row != 2:  #Take 2 first row
#         for curr_col in range(0, colcount , 1): #5300 column
#             data = sheet.cell_value(curr_row, curr_col) # Read the data in the current cell
#             row_data.append(data)
        
#         Data.append(row_data)
    
#     else: 
#         for curr_col in range(0, colcount, 1): # Take last row
#             Output.append(sheet.cell_value(curr_row,curr_col))

# Data will store like this Data = [[0,2,9,...],
#                                   [3,4,6,...]] 
# print(Data[0][0]) = 0
# print(Data[1][0]) = 3
# print(Data[0][1]) = 2
# print(Data[1][1]) = 4

for curr_col in range(0, colcount, 1): #5300 columns
    col_data = []
    col_data_out =[]
    for curr_row in range(0, rowcount, 1): #3 rows
        if curr_row != 2:
            data = sheet.cell_value(curr_row, curr_col) #take data from 1 single cell
            col_data.append(data)
        else:
            data_out = sheet.cell_value(curr_row, curr_col) #take data from 1 single cell
            # col_data_out.append(data_out)
            Output.append(data_out)
    
    Data.append(col_data)
            
# print(Output[0:4])
# print(torch.tensor(Data[0:4]))
# print(len(Data))

# Data will store like this Data = [[0,2],[3,4],....,[2,3]] 
# print(Data[0][0]) = 0
# print(Data[1][0]) = 2
# print(Data[0][1]) = 3
# print(Data[1][1]) = 4

