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
# # Preprosessering og Pipelines

# %%

# %%

# %% [markdown]
# ## Litt om pipelines i sklearn

# %% [markdown]
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

# %%

# %% [markdown]
# Fra medium.com: "En scikit-learn-pipeline er et kraftig verktøy som lenker sammen flere trinn med dataforbehandling og modellering til en enkelt, strømlinjeformet enhet."
#
# Pipelines tilrettelegger veldig elegant koding av ML arbeidsflyter, og sikrer at alle tilpasning skjer på treningsdata og at det er ingen "leaking" hvor testdata blir brukt til noen del av tilpasningen ved et uhell.
#
# Et par eksempel:
#
# <code>model = Pipeline([
#               ('imputer', SimpleImputer(strategy ='mean')),   # Step 1: Impute missing values
#               ('classifier', DecisionTreeClassifier())   # Step 2: decision tree classifier
# ])</code> \
# \
# <code>model_LR_second_order = Pipeline([
#                      ('scale', StandardScaler()),                          
#                      ('second_order_interactions', PolynomialFeatures(2)),  
#                     ('linear_regression', LinearRegression()) 
# ])</code>
#

# %% [markdown]
# ## Nyttige steger i en ML arbeidsflyt
#
# - GENERERING AV FEATURES. steger under viser mulige transformasjoner som kan gjøres, bare for å tenke om flere betraktninger
#
# <br>
#
# - estatte/imputere manglende verdier: `SimpleImputer()` 
# - lage indikator variabler: `OneHotEncoder()` 
# - lage nye variabler: text features `CountVectorizer()`, `TfidfVectorizer()` \
# høyere grad kombinasjoner av features i.e. X_i*X_j: `PolynomialFeatures()` 
# - skalering: `StandardScaler()` 
#

# %%

# %% [markdown]
# ## Referanse til Python sklearn pakka:
#
# https://scikit-learn.org/stable/api/index.html 
#
# også
#
# https://scikit-learn.org/stable/supervised_learning.html

# %%

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

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

# %%

# %% [markdown]
# # Prøve å tilpasse en modell til opprinnelige data

# %% [markdown]
# Så lenge måtte vi kaste bort alle kategoriske variabler, og alle punktene som hadde en manglende verdi. Hvordan kan vi håndtere disse?

# %%

housing=pd.read_csv(f'{data_sti}/CA_housing_1990.csv', sep=';')

# %%
housing.dtypes

# %%
housing.isna().sum()

# %%
X=housing.drop('median_house_value', axis=1)

# %%
# legge X_ til kolonnenavn for X-variabler

X.columns='X_'+X.columns
X.columns

# %%
y= housing.loc[:,'median_house_value'].values
y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# %%
model = RandomForestRegressor()
#model.fit(X_train, y_train)

# %% [markdown]
# # Håndtere kategoriske variabler
# ### Lage indikator variabler ('dummies')

# %%
enc = OneHotEncoder(sparse_output=False, drop='first')
ocean_dummies_train = enc.fit_transform(X_train[['X_ocean_proximity']])
dummies_tr=pd.DataFrame(ocean_dummies_train)
dummies_tr.columns = ['1','2','3','4']
dummies_tr.columns='X_ocean_prox_'+dummies_tr.columns

# %%
dummies_tr

# %%
enc.categories_

# %%
X_train=X_train.reset_index().drop(['index'], axis=1)
X_train=pd.concat([X_train, dummies_tr], axis=1).drop(['X_ocean_proximity'], axis=1)
X_train

# %%
ocean_dummies_test = enc.transform(X_test[['X_ocean_proximity']])
dummies_test=pd.DataFrame(ocean_dummies_test)
dummies_test.columns = ['1','2','3','4']
dummies_test.columns='X_ocean_prox_'+dummies_test.columns

X_test=X_test.reset_index().drop(['index'], axis=1)
X_test=pd.concat([X_test, dummies_test], axis=1).drop(['X_ocean_proximity'], axis=1)
X_test

# %% [markdown]
# # Håndtere manglende verdier

# %%
imp_mean = SimpleImputer(strategy='mean')
X_train_imp=imp_mean.fit_transform(X_train)
X_train_imp

# %%
sum(np.isnan(X_train_imp))

# %%
imp_mean.transform(X_test)

# %%

# %% [markdown]
# # Trene en modell med alle våre data i CA_housing_1990.csv

# %% [markdown]
# ### Håndtere manglende verdier

# %%
mymodel = Pipeline([   ('imputer', SimpleImputer(strategy ='mean')),
                       ('RandForest', RandomForestRegressor()) ])

# %%
mymodel.fit(X_train, y_train)

# sjekke MAE på testsettet
y_pred = mymodel.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Random forest, Mean Absolute Error: ${mae:.2f}')

# %%

# %% [markdown]
# # mer preprosessering: skalering og kvadratiske features

# %% [markdown]
# #### vi sammenligner enkel lineær regresjon og lineær regresjon med kvadratiske features og skalering

# %%

# %% [markdown]
# #### Enkel lineær regresjon

# %%
mymodel = Pipeline([   ('imputer', SimpleImputer(strategy ='mean')),
                       ('LinReg', LinearRegression()) ])

mymodel.fit(X_train, y_train)
y_pred = mymodel.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Simple linear regression, Mean Absolute Error: ${mae:.2f}')

# %%

# %% [markdown]
# #### Kvadratiske features

# %%
mymodel = Pipeline([   ('imputer', SimpleImputer(strategy ='mean')),
                       ('second_order_interactions', PolynomialFeatures(2)),
                       ('LinReg', LinearRegression()) ])

mymodel.fit(X_train, y_train)
y_pred = mymodel.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Quadratic features + linear regression, Mean Absolute Error: ${mae:.2f}')

# %%

# %% [markdown]
# #### Kvadratiske features + skalering

# %%
mymodel = Pipeline([   ('imputer', SimpleImputer(strategy ='mean')),
                       ('second_order_interactions', PolynomialFeatures(2)),
                       ('scale', StandardScaler()),            
                       ('LinReg', LinearRegression()) ])

mymodel.fit(X_train, y_train)
y_pred = mymodel.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Quadratic features + scaling + linear regression, Mean Absolute Error: ${mae:.2f}')

# %%

# %%

# %%
