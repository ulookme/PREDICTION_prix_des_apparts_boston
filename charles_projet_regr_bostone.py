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
df = pd.read_csv(r"C:\Users\utilisateur\Dowloads\dossierpython\boston_house_prices.csv")
# ou
# x, y = load_boston(return_X_y=True)

#Analyser les variables du dataset en affichant sa description
print(df.shape)
print(df.describe())

#Supprimer les valeurs nulles, s’il en existe.
df.isna().sum() #somme des na

# creation de  x 
x=np.array(df)
X = x[:, :-1]

# y target
y=np.array(df)
y = y[:, -1]

# Analyse de corrélation.
print(df.corr(method='pearson', min_periods=0))
print('\n')
print(df.corr(method='spearman'))

# Graph de correlation avec la methode Spearman
corrMatrix = df.corr(method='spearman')
mask = np.zeros_like(corrMatrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corrMatrix, mask=mask, vmax=.3, square=True, annot=True, center=0)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show(ax)
#selection des variables explicatives (features)ou nom des colonne les plus corréllé
X = X[:,(0,2,4,5,6,7,12)]

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print ("Train=" + str(X_train.shape) + ", Test=" + str(X_test.shape))

# Création du modèle de régression linéaire
reg = LinearRegression()

# Entrainement du modèle
reg.fit(X_train, y_train)

# Test du modèle
y_pred = reg.predict(X_test)

# Erreur quadratique moyenne (erreur d’estimation)
print('Erreur quadratique moyenne : %s'%mean_squared_error(y_test, y_pred))

# Coefficient
print('Coefficient : %s'%reg.coef_)

# R2
print('R2 : %s'%r2_score(y_test, y_pred))

# Graphiques
plt.plot(np.sort(y_test, axis=0))
plt.plot(np.sort(y_pred, axis=0), c='red')

