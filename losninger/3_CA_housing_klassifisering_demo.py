#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Klassifisering av boligverdier med "California housing" datasett

# %% [markdown]
#
#
#
#
#
# ## Litt om data: se eks. 7.2.7 "California housing data" på 
# https://scikit-learn.org/stable/datasets/real_world.html
#
# Bakgrunnsinfo om data fra den siden: \
# <br>
#     
# $Data Set Characteristics:$
#
# Number of Instances:
# 20640
#
# Number of Attributes:
# 8 numeric, predictive attributes \
# 1 categorical predictive attribute \
# and a categorical target variable (generated below)
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
# Target variable: \
# **house_value_categ** &emsp;&emsp;     categorical variable for the median value of a house in dollars - 
#

# %% [markdown]
# This dataset was obtained from the StatLib repository. https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
#
# The target variable is a categorical value for the median house value for California districts in dollars. It takes values .
#
# This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).
#
# A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

# %% [markdown]
#  **NB:** SSBs Dapla-system tillater ikke direkte nedlasting av datasetter fra eksterne kilder som ikke er inkludert som en del av sklearn-installasjonen. Følgende eksempelkode fra scikit-learn.org vil ikke fungere:
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale

import pickle


# %%
verbose=True


# %%
data_sti='../data'


# %%
pd.set_option('display.precision', 2)

# %% [markdown]
# ## Lese inn og forberede data

# %%
housing=pd.read_csv(f'{data_sti}/CA_housing_1990.csv', sep=';')

# samme bearbeiding som i regresjonseksempelet: 
# 1) fjerne kategorisk forklaringsvariabel ocean_proximity
# 2) fjerne punkter med manglende verdier
housing=housing.drop('ocean_proximity', axis=1)
housing = housing[~housing['average_bedrooms'].isna()]


# %%
housing


# %%
housing.hist(column='median_house_value', bins=50)


# %%
# vi lager en egendefinert kategorisk variabel som beskriver median_house_value som:
#            < 130000           =>    'Low'
#    130000 <= ... <= 240000    =>    'Medium'
#    240000 <  ... < 490000     =>    'High' 
#            >= 490000          =>    'Very pricy'

thresh_lowmed=130000
thresh_medhigh=240000
thresh_veryhigh=490000

def dollars_to_categ(house_vals):
    
    house_categs= np.select(
    [   house_vals<thresh_lowmed,
        house_vals.between(thresh_lowmed, thresh_medhigh, inclusive='both'), 
        house_vals.between(thresh_medhigh, thresh_veryhigh, inclusive='neither'),
        house_vals>=thresh_veryhigh   ], 
    [   'Low', 
        'Medium',
        'High',
        'Very pricy',
        ],
        default='unknown'
    )
    return(house_categs)
    
housing['house_value_categ']= dollars_to_categ(housing['median_house_value'])   
housing    


# %%
housing=housing.drop('median_house_value', axis=1)

# %% [markdown]
# ## Lage X, y 

# %%
X=housing.drop('house_value_categ', axis=1)


# %%
# legge X_ til kolonnenavn for X-variabler

X.columns='X_'+X.columns
X.columns


# %%
X


# %%
y= housing.loc[:,'house_value_categ'].values
y


# %%
# litt utforskning av data


# %%
X.shape


# %%
y.shape


# %%
# summary statistics
X.describe()


# %%
pd.DataFrame(y).value_counts()

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
# ## Trene modell 

# %%
# vi prøver først en support vector machine
model = SVC()
model.fit(X_train, y_train)

# %% [markdown]
# ### Litt inspisering
# Hvordan klassifiserer modellen et punkt med gjennomsnittlige verdier for alle X variabler?

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


# %%
# lage en kopi for senere
y_pred_SVM = y_pred

# %% [markdown]
# ### Vurdere modellen

# %%
# beregne nøyaktigheten (prosent av prediksjoner som er riktige) 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# %%
# beregne F1-score

# F1: geometric mean of recall and precision, 
#     weighted averaged over the categories for the outcome variable y
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"F1 score: {f1:.2f}")


# %%
# Vis 'classification report', som gir mer detaljerte mål
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# %%
# hvis ønsket, skriv ut klassifiserings rapport til en tekst fil:
print(report, file=open('classification_report_CA_housing_values_SVM.txt', 'w'))

# %% [markdown]
# ### Vurdere ytelsen i strata

# %%
# sjekke nøyaktigheten i strata hvor hus er nyere eller eldre
nyere = X_test['X_housing_median_age']<=20

accuracy_new = accuracy_score(y_test[nyere], y_pred[nyere])
print(f"Accuracy, housing_median_age<=20: {accuracy_new:.2f}")

# %% [markdown]
# ## Prøv andre algoritmer

# %%
# nå prøver vi en random forest
model=RandomForestClassifier()

# tilpasning, sjekk treningstid
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Model trained in {end_time-start_time:.1f} seconds")

# sjekke nøyaktigheten på testsettet
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# %%
# lage en kopi av modellen for lagring
model_RF=model 

# %% [markdown]
# ### Prøv med dine egne modelldefinisjoner her!

# %%
model = KNeighborsClassifier()

# tilpasning, sjekk treningstid
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Model trained in {end_time-start_time:.1f} seconds")

# sjekke nøyaktigheten på testsettet
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

model_KNN = model


# %%
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"F1 score: {f1:.2f}")


# %% [markdown]
# ## Støttefunksjoner for testing

# %%
# lag en funksjon for å sjekke nøyaktighet av en modell

def sjekke_noyaktighet(mymodel):
    mymodel.fit(X_train, y_train)
    y_pred = mymodel.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    print("Testing model ", mymodel, "\naccuracy=", round(accuracy,2))
    return None


# %%
# prøv logistisk regresjon
# øke maksimalt antall iterasjoner for å hjelpe med konvergens

mymodel=LogisticRegression(max_iter=1000)
sjekke_noyaktighet(mymodel)


# %% [markdown]
# ingen konvergens selv med den høyere maksimalt antallen iterasjoner\
# så prøver vi å skalere X i dataene først

# %%
mymodel = Pipeline([   ('scale', StandardScaler()),
                  ('LR', LogisticRegression(max_iter=1000)) ])
sjekke_noyaktighet(mymodel)

# nå har vi konvergens og bedre ytelse!

# %% [markdown]
# hjelper skalering med de andre algoritmene vi har prøvd også?

# %%
# support vector machine (SVM)

mymodel = Pipeline([   ('scale', StandardScaler()),
                  ('SVM', SVC()) ])
sjekke_noyaktighet(mymodel)


# %%
# random forest

mymodel = Pipeline([   ('scale', StandardScaler()),
                  ('RF', RandomForestClassifier()) ])
sjekke_noyaktighet(mymodel)


# %%





# %%





# %%





# %%




