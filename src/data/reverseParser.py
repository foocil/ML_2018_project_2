import csv
import numpy as np
def reverseParser(sparse_matrix, filename):
    with open(filename, 'w',newline='') as csvfile:
        linewriter = csv.writer(csvfile, delimiter =',')
        linewriter.writerow(["Id","Prediction"])
        for column in range(sparse_matrix.shape[1]):
            for row in range(sparse_matrix.shape[0]):
                if(sparse_matrix[row][column] > 0):
                    rx_cy = "r"+str(row+1)+"_c"+str(column+1)
                    linewriter.writerow([rx_cy, sparse_matrix[row][column]])