import csv
import numpy as np
from src.helper import compute_error_alternative

def MF_SGD(data):
    item_features, user_features, rmse = matrix_factorization_SGD(data, 0.008, 0.01, 0.01, 20, 30)
    sparse_matrix = user_features.dot(item_features.T)
    return sparse_matrix

def matrix_factorization_SGD(train, gamma, lambda_user, lambda_item, num_features, num_epochs):
    """matrix factorization by SGD."""
    errors = [0]
    
    # seed
    np.random.seed(988)  
    
    # init matrices
    num_users, num_items = train.shape 
    user_features = np.random.rand(num_users, num_features)
    item_features = np.random.rand(num_items, num_features)
    
    # non-zero ratings indices 
    nz_row, nz_col = np.nonzero(train)
    nz_train = list(zip(nz_row, nz_col))

    print("Learning matrix factorization...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
		
        # decrease step size
        gamma /= 1.2
        
        for x, y in nz_train:
            item_info = item_features[y]
            user_info = user_features[x]
			
            # compute error : correct value - prediction
            err = train[x, y] - user_info.dot(item_info.T)
			
            # update latent factors
            item_features[y] += gamma * (err * user_info - lambda_item * item_info)
            user_features[x] += gamma * (err * item_info - lambda_user * user_info)
			
        rmse = compute_error_alternative(train, user_features, item_features, nz_train)
        print("iteration: {}, RMSE on training set: {}.".format(it, rmse))
        errors.append(rmse)
		
    return item_features, user_features, rmse
