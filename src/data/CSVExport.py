import csv
import numpy as np

def csvexport(sparse_matrix, pathToFinal):
    ## given the sample submission, it will update the value of each line.
    
    with open('data/sample_submission.csv', 'r') as csvfile, open(pathToFinal, 'w', newline='') as csvFinal:
        linereader = csv.reader(csvfile, delimiter =',')
        linewriter = csv.writer(csvFinal, delimiter =',')
        linewriter.writerow(["Id","Prediction"])
        next(linereader)
        for row in linereader:
            rx, cy = row[0].split('_')
            x, y = int(rx[1:]), int(cy[1:])
            rx_cy = "r"+str(x)+"_c"+str(y)
            if(sparse_matrix[x-1][y-1] < 1): sparse_matrix[x-1][y-1] = 1
            if(sparse_matrix[x-1][y-1] > 5): sparse_matrix[x-1][y-1] = 5
            linewriter.writerow([rx_cy, int(round(sparse_matrix[x-1][y-1]))])
    print("Writing is done")
