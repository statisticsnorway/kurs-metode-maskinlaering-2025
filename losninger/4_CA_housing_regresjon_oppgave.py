# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# %% [markdown]
# # Regresjon av boligverdier for bolig ved vannkanten

# %% [markdown]
# #### Denne oppgaven bruker datafilen **CA_housing_waterfront_regr.csv**.
#
# Det er et utdrag av *CA_housing_1990.csv* dataene som var brukt i de demo filene, og inneholder bolig der 'ocean_proximity' variabel tok en av følgende verdier: (NEAR OCEAN, NEAR BAY, ISLAND).
#

# %% [markdown]
# ## Litt om data: se eks. 7.2.7 "California housing data" på 
# https://scikit-learn.org/stable/datasets/real_world.html
#
# Bakgrunnsinfo om data fra den siden: \
# <br>
#     
# $Data Set Characteristics:$
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
# ## Referanse til Python sklearn pakka:
#
# https://scikit-learn.org/stable/api/index.html 
#
# også
#
# https://scikit-learn.org/stable/supervised_learning.html

# %% [markdown] jupyter={"outputs_hidden": true}
# # Initialiseringer

# %%
import pandas as pd
import numpy as np
import dapla as dp
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

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lese inn data

# %%
waterfront=pd.read_csv(f'{data_sti}/CA_housing_waterfront_regr.csv', sep=';')

# %% [markdown]
# # Inspisere data

# %%
waterfront.shape

# %%
waterfront.dtypes

# %%
waterfront.isna().sum()

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lage X, y 

# %%
X=waterfront.drop('median_house_value', axis=1)

# %%
X.columns='X_'+X.columns
X.columns

# %%
X

# %%
y= waterfront.loc[:,'median_house_value'].values
y

# %% [markdown]
# # Sjekk beskrivende statistikk

# %%
X.describe()

# %%
pd.DataFrame(y).describe()

# %% [markdown] jupyter={"outputs_hidden": true}
# # Dele i trenings- og testsetter

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# %%
X_train.shape

# %%
y_train.shape

# %%
X_test.shape

# %% [markdown] jupyter={"outputs_hidden": true}
# # Trene random forest modellen

# %%
# nå prøver vi en ekte ML-s metode, en tilfeldig skog (random forest) 
model = RandomForestRegressor()

# tilpasning, sjekk treningstid
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Training time: {end_time-start_time:.1f} seconds")

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lage prediksjoner i testsettet

# %%
y_pred = model.predict(X_test)

# %%

# %% [markdown]
# # Vurdere modellen

# %%
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: ${mae:.2f}')

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Hyperparameter justering (tuning)

# %% [markdown]
# **NB**: Kryss-validering kan ta lenge! Som standard bruker GridSearchCV(...) 5-fold kryssvalidering og trener 5 modeller for hver kombinasjon av parametere som er prøvd.

# %%
CV_RF = GridSearchCV(RandomForestRegressor(), 
                     param_grid={'min_samples_leaf': [1,2,3]}, 
                     scoring='neg_mean_absolute_error')

# fit-funksjonen her splitter opp treningssettet i trenings- og valideringssetter, 
#    5 separate ganger, og vurderer ytelsen over de 5 tilfellene 
CV_RF.fit(X_train, y_train)

# %%

# %% [markdown]
# # Finn beste innstillingen av parameteren

# %%
CV_RF.cv_results_

# %%
CV_RF.best_estimator_

# %% [markdown]
# # Lage prediksjoner i testsettet med den beste innstillingen

# %%
y_pred=CV_RF.predict(X_test)

# %%

# %% [markdown]
# # Sjekk ytelsen med den beste innstillingen

# %%
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: ${mae:.2f}')

# %%

# %%

# %%
