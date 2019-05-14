import csv
import numpy as np


def parser(path_csv) :
    maxUserId = 0
    maxMovieId = 0
    with open(path_csv, 'r') as csvfile:
        linereader = csv.reader(csvfile, delimiter = ',')
        next(linereader)
        
        for row in linereader:
            rx, cy = row[0].split('_')
            x, y = int(rx[1:]), int(cy[1:])
            maxUserId = max(maxUserId, x)
            maxMovieId = max(maxMovieId, y)
    sparse_matrix = np.zeros((maxUserId,maxMovieId))
    with open(path_csv, 'r') as csvfile:
        linereader = csv.reader(csvfile, delimiter = ',')
        next(linereader)
        for row in linereader:
            rx, cy = row[0].split('_')
            x, y = int(rx[1:]), int(cy[1:])
            sparse_matrix[x-1][y-1] = int(row[1])
    return sparse_matrix