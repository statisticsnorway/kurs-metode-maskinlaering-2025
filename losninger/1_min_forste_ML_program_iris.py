#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Eksempel 1: klassifisering av "Iris" datasett

# %%
# Importere python moduler
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# %% [markdown]
# # Laste inn data
# For flere detaljer se på: https://scikit-learn.org/stable/datasets/toy_dataset.html

# %%
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# %% [markdown]
# # Inspisere data

# %%
# X verdier
df = pd.DataFrame(data=X, columns=iris.feature_names)
df


# %%
y


# %%
df['y']=y
df


# %%
iris.target_names

# %% [markdown]
# # Dele i trenings- og testsetter

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# %%
len(X_train)


# %%
len(X_test)


# %%
len(y_train)


# %%
len(y_test)


# %%
X_train[:6]

# %% [markdown]
# # Tilpasse en modell

# %%
model = LogisticRegression(max_iter=1000)  # max_iter parameter kan justeres

# Trene / tilpasse modellen på treningsdata
model.fit(X_train, y_train)


# %% [markdown]
# # Lage prediksjoner

# %%
# lage prediksjoner på test data
y_pred = model.predict(X_test)
y_pred


# %% [markdown]
# # Sjekk ytelsen

# %%
# beregne nøyaktighet av modellen på testdata
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# %%
# Vise 'classification report', som gir mer detaljerte mål
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classification Report:\n", report)


# %%





# %%




