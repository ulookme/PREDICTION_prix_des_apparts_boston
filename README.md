# prediction_prix_des_appart_bostone
prédire le prix des appartements database  Boston 
library
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#IMPORTATION DES DONNEES

#Analyser les variables du dataset en affichant sa description

#Supprimer les valeurs nulles, s’il en existe.

#Analyse de corrélation.

#selection des variables explicatives (features)ou nom des colonne les plus corréllé

#Split the data into training/testing sets

# Création du modèle de régression linéaire

# Entrainement du modèle

# Test du modèle

#Erreur quadratique moyenne (erreur d’estimation)
Coefficient
R2

