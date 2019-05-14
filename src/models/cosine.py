#########################
#   Cosine Similarity   #
#         for           #
# Recommandation System #
#########################

import sys
sys.path.append('../data/')
import src.data.reverseParser as csv

import numpy as np
import pandas as pd
import scipy.sparse as sc
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import multiprocessing
from joblib import Parallel, delayed

#########################################
# hard_coded values                     #
# can/must be tuned to reach best score #
#########################################
metric='cosine'
k=3
#########################################

def cosine(matrix, sample, mode="users"):
    print("Cosine Similarity Model")
    M = np.asarray(matrix)
    M = pd.DataFrame(M)
    #result = np.zeros((M.shape[0], M.shape[1]))
    result = sc.lil_matrix(sample)
    irange = range(1, M.shape[0]+1)
    
    print("Computing cosine similatiry matrix")
    
    num_cores = multiprocessing.cpu_count()
    
    if (mode=="users"):
        print("Collaborative Filtering {} Based".format(mode[:-1]))
        Parallel(n_jobs=num_cores-1, require='sharedmem')(
            delayed(processUserBased)(M, result, metric, k, i) for i in irange)
    elif (mode=="items"):
        Parallel(n_jobs=num_cores-1, require='sharedmem')(
            delayed(processItemBased)(M, result, metric, k, i) for i in irange)
    else:
        print("ERROR - unknown mode")
        return
    
    print("Done with cosine function")
    
    return result
    
def processUserBased(M, result, metric, k, i):
    for j in range(1, M.shape[1]+1):
        if (M.iloc[i-1, j-1] != 0.):
            result[i-1, j-1] = M.iloc[i-1, j-1]
        elif (result[i-1, j-1] != 0.):
            result[i-1, j-1] = predictBasedOnUser(i, j, M, metric, k)
            
def processItemBased(M, result, metric, k, i):
    for j in range(1, M.shape[1]+1):
        if (M.iloc[i-1, j-1] != 0.):
            result[i-1, j-1] = M.iloc[i-1, j-1]
        elif (result[i-1, j-1] != 0.):
            result[i-1, j-1] = predictBasedOnItem(i, j, M, metric, k)
    
def findksimilarusers(user_id, ratings, metric = metric, k=k):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
            
    return similarities,indices

def findksimilaritems(item_id, ratings, metric = metric, k = k):
    similarities = []
    indices = []
    model_kni = NearestNeighbors(metric = metric, algorithm='brute')
    model_kni.fit(ratings)
    
    distances, indices = model_kni.kneighbors(ratings.iloc[item_id-1, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1-distances.flatten()
    
    return similarities, indices
        
def predictBasedOnUser(user_id, item_id, ratings, metric=metric, k=k):
    prediction=0
    similarities, indices = findksimilarusers(user_id, ratings, metric, k)
    
    mean_rating = ratings.loc[user_id-1,:].mean()
    sum_similarities = np.sum(similarities)-1
    product = 1
    wtd_sum = 0
    flatIndices = indices.flatten()
    
    for i in range(0, len(flatIndices)):
        if flatIndices[i]+1 == user_id:
            continue
        else:
            ratings_diff = ratings.iloc[flatIndices[i],item_id-1]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
        prediction = int(round(mean_rating + (wtd_sum/sum_similarities)))
    return prediction
                        
def predictBasedOnItem(user_id, item_id, ratings, metric=metric, k=k):
    prediction = wtd_sum = 0
    similarities, indices=findksimilaritems(item_id, ratings.T)
    sum_wt = np.sum(similarities)-1
    product = 1
    flatIndices = indices.flatten()
    
    for i in range(0, len(flatIndices)):
        if flatIndices[i]+1 == item_id:
            continue
        else:
            product = ratings.iloc[user_id-1, flatIndices[i]]*(similarities[i])
            wtd_sum += product
    prediction = int(round(wtd_sum/sum_wt))
    return prediction
