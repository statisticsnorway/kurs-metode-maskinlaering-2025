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
import numpy as np
import dapla as dp
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
waterfront=pd.read_csv(f'{data_sti}/CA_housing_waterfront_classif.csv', sep=';')

# %% [markdown]
# # Inspisere data

# %%
waterfront.shape

# %%
waterfront.dtypes

# %%
waterfront.isna().sum()

# %%
waterfront.head()

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lage X, y 

# %%
X=waterfront.drop('house_value_categ', axis=1)

# %%
X.columns='X_'+X.columns
X.columns

# %%
X

# %%
y= waterfront.loc[:,'house_value_categ'].values
y

# %%

# %% [markdown]
# # Sjekk beskrivende statistikk

# %%
X.describe()

# %%
pd.DataFrame(y).value_counts()

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
# # Trene support vector machine modellen

# %%
model = SVC()
model.fit(X_train, y_train)

# %%

# %% [markdown] jupyter={"outputs_hidden": true}
# # Lage prediksjoner i testsettet

# %%
y_pred = model.predict(X_test)


# %%
y_pred_SVM = y_pred

# %% [markdown]
# # Vurdere modellen

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# %%
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"F1 score: {f1:.2f}")

# %% [markdown]
# # Skriv ut en klassifiseringsrapport

# %%
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# %%

 # %%
 labels=['Very pricy', 'High', 'Medium', 'Low']

# %%
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
conf_matrix

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
disp.plot()

# %%
pd.DataFrame(y_test).value_counts()

# %%
