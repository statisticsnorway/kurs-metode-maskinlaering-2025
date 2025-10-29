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
# # Klassifisering av boligverdier for bolig ved vannkanten

# %% [markdown]
# #### Denne oppgaven bruker datafilen **CA_housing_waterfront_classif.csv**.
#
# Det er et utdrag av *CA_housing_1990.csv* dataene som var brukt i de demo filene, og inneholder boliger der 'ocean_proximity' variabel tok en av følgende verdier: (NEAR OCEAN, NEAR BAY, ISLAND).
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
# **house_value_categ** &emsp;&emsp;     categorical variable for the median value of a house in dollars 
#

# %% [markdown]
# ## Referanse til Python sklearn pakka:
#
# https://scikit-learn.org/stable/api/index.html 
#
# også
#
# https://scikit-learn.org/stable/supervised_learning.html

# %% [markdown]
# # Initialiseringer

# %%
import pandas as pd
import dapla as dp
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
data_sti='../data/'
modell_sti='../modeller/'
resultat_sti='../resultater/'

# %%
pd.set_option('display.precision', 2)

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lese inn data

# %%

# %% [markdown]
# # Inspisere data

# %%

# %%

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lage X, y 

# %%

# %%

# %%

# %% [markdown]
# # Sjekk beskrivende statistikk

# %%

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Dele i trenings- og testsetter

# %%

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Trene support vector machine modellen

# %%

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lage prediksjoner i testsettet

# %%

# %%

# %% [markdown]
# # Vurdere modellen

# %%

# %%

# %% [markdown]
# # Skriv ut en klassifiseringsrapport

# %%

# %%

# %%
