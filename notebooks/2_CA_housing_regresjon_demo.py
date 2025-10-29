#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Regresjon av boligverdier med "California housing" datasett
#
# ## Litt om data: se eks. 7.2.7 "California housing data" på 
# https://scikit-learn.org/stable/datasets/real_world.html
#
# Bakgrunnsinfo om data fra den siden: 
#     
# $Data Set Characteristics:$
#
# Number of Instances:
# 20640
#
# Number of Attributes:
# 8 numeric, predictive attributes \
# 1 categorical predictive attribute \
# and a numeric target variable
#
# Attribute Information: 
#
# **longitude**  &emsp;    block group longitude \
# **latitude**   &emsp; &emsp;   block group latitude  \
# **housing_median_age**   &emsp;     median house age in block group \
# **average_rooms**    &emsp;   average number of rooms per household \
# **average_bedrooms**  &emsp;   average number of bedrooms per household \
# **population**  &emsp;   block group population \
# **average_occupancy**    &emsp;   average number of household members \
# **median_income**    &emsp;&emsp;     median income in block group \
# **ocean_proximity** &emsp;&emsp;     promixity to ocean ["'NEAR BAY'", "'1H OCEAN'", 'INLAND', "'NEAR OCEAN'", 'ISLAND']
#
#
# Target variable: \
# **median_house_value** &emsp;&emsp;     median value of a house in dollars
#

# %% [markdown]
# This dataset was obtained from the StatLib repository. https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
#
# The target variable is the median house value for California districts in dollars.
#
# This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).
#
# A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

# %% [markdown]
# **NB:** SSBs Dapla-system tillater ikke direkte nedlasting av datasetter fra eksterne kilder som ikke er inkludert som en del av sklearn-installasjonen. Følgende eksempelkode fra scikit-learn.org vil ikke fungere:
#
# - from sklearn.datasets import fetch_openml 
# - california_housing = fetch_openml(data_id='43939', parser="auto")
#
# Dette datasettet var lastet ned fra openML.org utenfor Dapla, fikk noe parsing og redigering, og ble kopiert til Dapla.
#
# ## Referanse til Python sklearn pakka:
#
# https://scikit-learn.org/stable/api/index.html 
#
# også
#
# https://scikit-learn.org/stable/supervised_learning.html

# %% [markdown]
# ## Initialiseringer

# %%
import pandas as pd
import numpy as np
import time
import math

import statistics
import sys


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

import pickle


# %%
verbose=True


# %%
data_sti='../data'
modell_sti='../modeller'
resultat_sti='../resultater'


# %%
pd.set_option('display.precision', 2)


# %%
import sklearn 
print(sklearn.__version__)

# %% [markdown]
# ## Lese inn og inspisere data

# %%

housing=pd.read_csv(f'{data_sti}/CA_housing_1990.csv', sep=';')


# %%
housing


# %%
housing.dtypes


# %%
# vi skal ikke håndtere kategoriske forklaringsvariabler i dagens modulen, 
#   så tar vi ut ocean_proximity
housing=housing.drop('ocean_proximity', axis=1)


# %%
# sjekke om vi har manglende verdier, NaN eller None
housing.isna().sum()


# %%
# vi skal ikke håndtere manglende verdier i dagens modulen,
#   så tar vi ut de 207 punktene med manglende average_bedrooms
housing = housing[~housing['average_bedrooms'].isna()]


# %%
housing.isna().sum()


# %%
housing

# %% [markdown]
# ## Lage X, y 

# %%
X=housing.drop('median_house_value', axis=1)


# %%
# legge X_ til kolonnenavn for X-variabler
X.columns='X_'+X.columns
X.columns


# %%
X


# %%
y= housing.loc[:,'median_house_value'].values
y


# %%
# litt utforskning av data
X.shape


# %%
y.shape


# %%
# summary statistics
X.describe()


# %%
pd.DataFrame(y).describe()


# %% [markdown]
# ## Dele i trenings- og testsetter

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)


# %%
X_train.shape


# %%
X_test.shape


# %%
y_train.shape


# %%
y_test.shape

# %% [markdown]
# ## Trene modellen

# %%
# begynner med enkelst mulig eksempel, linear regresjon
model = LinearRegression()
model.fit(X_train, y_train)

# %% [markdown]
# ### Litt inspisering
# vi kan inspisere modellen i forskjellige måter, f.eks. vi kan sjekke
#  koeffisientene av regresjonsmodellen

# %%
model.coef_


# %%
X.columns

# %% [markdown]
# hva predikerer modellen for et punkt med gjennomsnittlige verdier for alle X variabler?

# %%
means = X.describe().loc[['mean']]
means


# %%
model.predict(means)


# %% [markdown]
# ## Test modellen

# %%
# lage prediksjoner på testdata
y_pred = model.predict(X_test)
y_pred

# %% [markdown]
# ### Vurdere modellen

# %%
# skriv ut root mean squared (RMS) feil:
root_mse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: ${root_mse:.2f}')

# skriv ut mean absolute error (MAE):
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: ${mae:.2f}')

# R^2 gir andelen av varians i utgangsvariabel y som kan predikeres fra 
#    forklaringsvariablene X i en linear modell
r2 = r2_score(y_test, y_pred)
print(f'R^2: ${r2:.2f}')

# %% [markdown]
# ### Vurdere ytelsen i strata

# %%
# sjekke MAE i strata hvor hus er nyere eller eldre
nyere = X_test['X_housing_median_age']<=20

mae_ny = mean_absolute_error(y_test[nyere], y_pred[nyere])
print(f'Mean Absolute Error, housing_median_age<=20: ${mae_ny:.2f}')

mae_gammel = mean_absolute_error(y_test[~nyere], y_pred[~nyere])
print(f'Mean Absolute Error, housing_median_age>20: ${mae_gammel:.2f}')

# %% [markdown]
# ## Prøv andre algoritmer

# %%
# nå prøver vi en ekte ML-s metode, en tilfeldig skog (random forest) 
model = RandomForestRegressor()

# tilpasning, sjekk treningstid
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Training time: {end_time-start_time:.1f} seconds")

# sjekke MAE på testsettet
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: ${mae:.2f}')


# %%
# lage en kopi av modellen og prediksjoner for lagring
model_RF=model 
y_pred_RF= y_pred


# %% [markdown]
# ### Prøv med dine egne modelldefinisjoner her!

# %%

# %%

# %%

# %% [markdown]
# ## Hyperparameter justering (tuning)

# %%
# lag en funksjon for å sjekke MAE ytelsen av en modell
def sjekke_MAE(mymodel):
    mymodel.fit(X_train, y_train)
    y_pred = mymodel.predict(X_test)
    MAE=mean_absolute_error(y_test, y_pred)
    print("Testing model ", mymodel, "\nMAE=$", round(MAE,2))
    return(round(MAE,2))


# %% [markdown]
# <code>RandomForestRegressor()</code> bruker 100 trær som standard. Hva om vi endrer det?

# %%
RF_100trees=RandomForestRegressor(n_estimators = 100)
sjekke_MAE(RF_100trees)


# %%
RF_50trees=RandomForestRegressor(n_estimators = 50)
sjekke_MAE(RF_50trees)

# %% [markdown]
# #### Bruk kryss-validering! 
# <code>GridSearchCV(...)</code> funksjonen nedenfor gjør alt arbeidet for deg,
# og tester med 5-gangers kryssvalidering på hver av de 3 spesifiserte innstillingene av
# hyperparametre. 
#
# Først sjekk spesifikasjonen av <code>RandomForestRegressor(...)</code> funksjonen på:\
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html \
# Vi ser at RandomForest(...) har er en *n_estimators* hyperparameter som kontroller antall trær. Vi spesifiserer nedenfor i <code>GridSearchCV(...)</code> funksjonen at RandomForestRegressor(...) skal kjøres med *n_estimators* hyperparameteren satt til verdiene 50, 100, og 200, en etter en. Vi spesifiserer også at mean absolute error skal brukes for å vurdere ytelse.
#
# **NB**: Kryss-validering kan ta lenge! Koden nedenfor trener 15 modeller.

# %%
CV_RF = GridSearchCV(RandomForestRegressor(), 
                     param_grid={'n_estimators': [50,100,200]}, 
                     scoring='neg_mean_absolute_error')

# fit-funksjonen her splitter opp treningssettet i trenings- og valideringssetter, 
#    5 separate ganger, og vurderer ytelsen over de 5 tilfellene 
CV_RF.fit(X_train, y_train)


# %%
CV_RF.cv_results_

# %% [markdown]
# Sjekk hvilken innstilling av parametere hadde den beste ytelsen. 
# Den innstillingen er merkert og brukt når vi kaller predict-funksjonen.

# %%
CV_RF.best_estimator_


# %%
# sjekk ytelse 
y_pred=CV_RF.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: ${mae:.2f}')

# %% [markdown]
# ## Lage en prediksjonsmatrise
# ... med X, y verdier, prediksjoner,og mål på feil på testsettet

# %%
predic_matrix= X_test.copy()
predic_matrix.insert(len(X_test.columns),'median_dollar_value_true', y_test)

predic_matrix['median_dollar_value_pred_RandFor']=y_pred_RF
predic_matrix['RandFor_error']= y_pred_RF - y_test

predic_matrix

# %% [markdown]
# # Lagre modellene og prediksjonene
#
# ### Lagre modellen
#
# lagre modellen på disc, så vi kan bruke den for predikering senere

# %%
# lagre modellen på disk
filename = f'{modell_sti}/California_housing_values_RandForest_model.sav'
pickle.dump(model_RF, open(filename, 'wb'))

# %% [markdown]
# en gang senere... \
# vi kan bare laste inn modellen og predikere med en gang, uten modellbygging

# %%
loaded_model = pickle.load(open(filename, 'rb'))

# %% [markdown]
# teste predikering på den lastede modellen

# %%
newX = X_test.iloc[[1]]
newX


# %%
loaded_model.predict(newX)


# %%
model_RF.predict(newX)

# %% [markdown]
# ### Lagre prediksjonsmatrisen

# %%
predic_matrix.to_csv(f'{resultat_sti}/California_housing_values_RandFor_predictions.csv', float_format='%.2f', sep=';', index=False)


# %%





# %%





# %%




