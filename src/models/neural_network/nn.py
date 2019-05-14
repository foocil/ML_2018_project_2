from abc import ABC, abstractmethod
import csv
import pandas as pd
import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.constraints import non_neg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class Neural(ABC):
    """
    Abstract class for a Neural Network
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.history = None
        self.name = None
        self.predictions = None
        self.report_name = None
        self.report_path = None
        self.nonneg=False
        
    def summary(self):
        print(self.model.summary())        
        
    @abstractmethod
    def create_model(self, n_users, n_items):
        """
        Creates and compile the model
        
        Parameters
        ----------
        n_users : int
            The number of users
        n_items : int
            The number of items
        """
        pass
    
    @abstractmethod
    def train(self, df, epochs):
        """
        Trains (fit) the model
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the training data
        epochs: int
            The number of epochs for the training
        """
        pass
    
    def predict(self, user_id, item_id):
        """
        Performs the prediction for given data (from a test DataFrame)
        
        Parameters
        ----------
        user_id : list(int)
            List of users
        item_id : list(int)
            List of items
        """
        print("predicting...")
        self.predictions = self.model.predict([user_id, item_id], verbose=1)
        self.predictions = np.round(self.predictions)
        for i in range(len(self.predictions)):
            if (self.predictions[i] < 1):
                self.predictions[i] = 1
            if (self.predictions[i] > 5):
                self.predictions[i] = 5
        print("done predicting")
    
    def save(self, subdf, path=None):
        """
        Saves the predictions to a file
        
        Parameters
        ----------
        subdf : pandas.DataFrame
            DataFrame containing the item and users used for the prediction
        path : str
            Path to save the file (with extension)
        """
        if (path is not None):
            self.report_path = path
            
        print("saving predictions...", end='\r')
        subdf.rating = self.predictions
        with open(self.report_path, 'w', newline='') as out:
            linewriter = csv.writer(out, delimiter =',')
            linewriter.writerow(["Id","Prediction"])
            for i in range(len(subdf)):
                rx_cy = "r"+str(subdf.user_id[i]+1)+"_c"+str(subdf.item_id[i]+1)
                linewriter.writerow([rx_cy, int(subdf.rating[i])])
            print("prediction saved at {}".format(self.report_path))
            
            
class NeuralCF(Neural):
    """Collaborative Filtering Neural Network"""
    
    def __init__(self, nonneg=False):
        Neural.__init__(self)
        self.nonneg = nonneg
        
    def create_model(self, n_users, n_items):
        user_id_input = keras.layers.Input(shape=[1], name='user')
        item_id_input = keras.layers.Input(shape=[1], name='item')

        user_embedding = keras.layers.Embedding(output_dim=100,
                                                input_dim=n_users + 1,
                                                input_length=1,
                                                name='user_embedding')(user_id_input)
        if (self.nonneg):
            user_embedding = keras.layers.Embedding(output_dim=100,
                                                    input_dim=n_users + 1,
                                                    input_length=1,
                                                    name='user_embedding',
                                                    embeddings_constraint=non_neg())(user_id_input)

        item_embedding = keras.layers.Embedding(output_dim=100,
                                                input_dim=n_items + 1,
                                                input_length=1,
                                                name='item_embedding')(item_id_input)
        if (self.nonneg):
            item_embedding = keras.layers.Embedding(output_dim=100,
                                                    input_dim=n_items + 1,
                                                    input_length=1,
                                                    name='item_embedding',
                                                    embeddings_constraint=non_neg())(item_id_input)

        user_vecs = keras.layers.Flatten()(user_embedding)
        item_vecs = keras.layers.Flatten()(item_embedding)

        user_dropout = keras.layers.Dropout(0.2, name="user_dropout")(user_vecs)
        item_dropout = keras.layers.Dropout(0.2, name="item_dropout")(item_vecs)

        y = keras.layers.dot([user_dropout, item_dropout], axes=1)

        self.model = keras.models.Model(inputs=[user_id_input, item_id_input], outputs=[y])
        self.model.compile(optimizer='adam', loss='mae')
        
    def train(self, df, epochs=10):
        train, test = train_test_split(df, test_size=0.2)
        self.report_name = "neural_cf{}_{}".format("_nonneg" if self.nonneg else "", epochs)
        self.report_path = 'reports/to_submit/{}.csv'.format(self.report_name)
        
        self.history = self.model.fit([df.user_id, df.item_id],
                                      df.rating,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=([test.user_id, test.item_id], test.rating))
        
        y_hat = np.round(self.model.predict([test.user_id, test.item_id]), 0)
        mae = mean_absolute_error(y_hat, test.rating)
        print("Training done with a loss of {}".format(mae))
        
        
class NeuralFactor(Neural):
    """Matrix Factorization Neural Network"""
    
    def __init__(self, n_latent_users, n_latent_items, nonneg=False):
        Neural.__init__(self)
        self.n_latent_factors_user = n_latent_users
        self.n_latent_factors_movie = n_latent_items
        self.nonneg = nonneg
        
    def create_model(self, n_users, n_items):
        movie_input = keras.layers.Input(shape=[1],name='Item')
        movie_embedding = keras.layers.Embedding(n_items + 1, self.n_latent_factors_movie, name='Movie-Embedding')(movie_input)
        if (self.nonneg):
            movie_embedding = keras.layers.Embedding(n_items + 1, self.n_latent_factors_movie, name='Movie-Embedding', embeddings_constraint=non_neg())(movie_input)
        movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
        movie_vec = keras.layers.Dropout(0.2)(movie_vec)

        user_input = keras.layers.Input(shape=[1],name='User')
        user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, self.n_latent_factors_user,name='User-Embedding')(user_input))
        if (self.nonneg):
            user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, self.n_latent_factors_user,name='User-Embedding', embeddings_constraint=non_neg())(user_input))
        user_vec = keras.layers.Dropout(0.2)(user_vec)

        concat = keras.layers.concatenate([movie_vec, user_vec], name='Concat')
        concat_dropout = keras.layers.Dropout(0.2)(concat)
        dense = keras.layers.Dense(200,name='FullyConnected')(concat)
        dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
        dense_2 = keras.layers.Dense(100,name='FullyConnected-1')(concat)
        dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
        dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dense_2)
        dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
        dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)

        result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
        adam = Adam(lr=0.005)
        self.model = keras.Model([user_input, movie_input], result)
        self.model.compile(optimizer=adam,loss= 'mean_absolute_error')
        
        
    def train(self, df, epochs=100):
        train, test = train_test_split(df, test_size=0.2)
        self.report_name = "neural_factor_{}_{}{}_{}".format(self.n_latent_factors_user,
                                                             self.n_latent_factors_movie,
                                                             "_nonneg" if self.nonneg else "",
                                                              epochs)
        self.report_path = 'reports/to_submit/{}.csv'.format(self.report_name)
        
        self.history = self.model.fit([train.user_id, train.item_id],
                                      train.rating,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=([test.user_id, test.item_id], test.rating))
        
        y_hat = np.round(self.model.predict([test.user_id, test.item_id]), 0)
        mae = mean_absolute_error(y_hat, test.rating)
        print("Training done with a loss of {}".format(mae))
        
def createDataFrameCSV(source, output):
    """
    Creates a DataFrame-like csv file with the data from a raw csv
    
    Parameters
    ----------
    source : str
        Path to the raw data csv file
    output : str
        Path to the output DataFrame-like csv file
    """
    with open(source, 'r', newline='') as file, open(output, 'w', newline='') as out:
        linereader = csv.reader(file, delimiter = ',')
        linewriter = csv.writer(out, delimiter =',')
        linewriter.writerow(["user_id","item_id","rating"])
        next(linereader)
        
        for row in linereader:
            ids = row[0].split('_')
            rating = row[1]
            user_id = ids[0][1:]
            item_id = ids[1][1:]
            linewriter.writerow([user_id, item_id, rating])