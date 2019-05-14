#########################
#    Neural Network     #
#         for           #
# Recommandation System #
#########################

import sys
sys.path.append('../data/')
import src.data.reverseParser as csv
import click
from src.models.neural_network import nn
import numpy as np
import pandas as pd
import os.path

@click.command()
@click.option("--tpe", "-t",
              prompt=True,
              type=click.Choice(["factorization", "cf", "all"]),
              default="factorization",
              help="Type of Neural Network")
@click.option("--epochs", "-e",
              prompt=True,
              default=100,
              help="Epoches for training")
@click.option("--nonneg", "-n",
              prompt=True,
              default=False,
              help="Add non-negativity constraint")
def neural(tpe, epochs, nonneg):
    """
    Launches neural network training and predictions
    """
    
    # checking DataFrame csv (and creating them is they don't exist)
    click.echo("checking data...")
    if (not os.path.exists("data/data_frame.csv")):
        click.echo("File 'data_frame.csv' does not exists")
        click.echo("Creating file...")
        nn.createDataFrameCSV("data/data_train.csv", "data/data_frame.csv")
        click.echo("File 'data_frame.csv' successfully created.")
        
    if (not os.path.exists("data/sample_dataFrame.csv")):
        click.echo("File 'sample_dataFrame.csv' does not exists")
        click.echo("Creating file...")
        nn.createDataFrameCSV("data/sample_submission.csv", "data/sample_dataFrame.csv")
        click.echo("File 'sample_dataFrame.csv' successfully created.")
        
    # loading DataFrame csv
    click.echo("loading data...")
    dd = pd.read_csv("data/data_frame.csv", sep=',')
    dd.user_id = dd.user_id.astype('category').cat.codes.values
    dd.item_id = dd.item_id.astype('category').cat.codes.values
    
    subdf = pd.read_csv('data/sample_dataFrame.csv')
    subdf.user_id = subdf.user_id.astype('category').cat.codes.values
    subdf.item_id = subdf.item_id.astype('category').cat.codes.values
    
    n_users = len(dd.user_id.unique())
    n_items = len(dd.item_id.unique())
    
    print()
    
    do_all = False
    
    # running model(s)
    if (tpe == "all"):
        click.echo("Start with factorization:")
        tpe = "factorization"
        do_all = True
    
    if (tpe == "factorization"):
        nlu = click.prompt("N latent factor user", default=5)
        nli = click.prompt("N latent factor item", default=8)
        model = nn.NeuralFactor(nlu, nli, nonneg)
    elif (tpe == "cf"):
        model = nn.NeuralCF(nonneg)
    
    model.create_model(n_users, n_items)
    model.summary()
    
    model.train(dd, epochs)
    model.predict(subdf.user_id, subdf.item_id)
    model.save(subdf, "reports/aaaaa.csv")
    
    print()
    
    if (do_all):
        click.echo("Now let's to the collaborative filtering")
        nonneg = click.prompt("Add non-negativity constraint?", default=False)
        epochs = click.prompg("Number of epochs:", default=10)
        model = nn.NeuralCF(nonneg)
        model.create_model(n_users, n_items)
        model.summary()

        model.train(dd, epochs)
        model.predict(subdf.user_id, subdf.item_id)
        model.save(subdf, "reports/bbbbb.csv")
