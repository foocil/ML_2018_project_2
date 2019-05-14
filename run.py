#!/usr/bin/env python

################################
# Machine Learning - Project 2 #
# Recommandation System        #
# Main Launcher                #
################################

import click

import sys
sys.path.append('/src/models')
sys.path.append('/src/data')
from src.models.cosine import cosine
from src.models.neural import neural
from src.data.parser import parser
from src.models.SGD_factorization import MF_SGD
from src.models.ALS import ALS
from src.data.parser import parser
from src.data.CSVExport import csvexport

############################
# Global variables
############################
dataTrainFileName = 'data/data_train.csv'
sampleSubmissionFileName = 'data/sample_submission.csv'

#######################
# Main Function
#######################

# click.options to be able to tune functions as we run them
@click.command()
@click.option("--action", "-a",
              prompt=True,
              type=click.Choice(["all", "als", "cosine", "neural", "sgd"]),
              default="als",
              help="Action to be performed")
def main(action):
    if (action != "neural"):
        print("Loading data...")
        dataTrainMatrix = parser(dataTrainFileName)
        sampleSubmission = parser(sampleSubmissionFileName)
        print()
    
    if (action == "cosine"):
        predicted_user = cosine(dataTrainMatrix, sampleSubmission, mode="users")
        predicted_item = cosine(dataTrainMatrix, sampleSubmission, mode="items")
        csvexport(predicted_user, "reports/cosine_user.csv")
        csvexport(predicted_item, "reports/cosine_item.csv")
    elif (action == "als"):
        predicted, rmse = ALS(dataTrainMatrix)
        csvexport(predicted.T, "reports/ALSMatrix.csv")
    elif (action == "sgd"):
        sparse_matrix = MF_SGD(dataTrainMatrix)
        csvexport(sparse_matrix, "SGD_submission.csv")
    elif (action =="neural"):
        neural()
    elif (action == "all"):
        cosine()
        als()
        sgd()
        neural()
        

if (__name__ == "__main__"):
    main()
