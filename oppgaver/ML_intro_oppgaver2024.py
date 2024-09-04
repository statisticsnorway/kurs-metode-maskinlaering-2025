#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Klassifisering av iris data
#
# #### Bruk *1_min_forste_ML_program_iris.py*.
#
# 1) **Se på trening / test split**: Kjør test/ training split med 75% i testdata (nå er det 25% i testdata). Hva er weighted average precision med den nye innstillingen?
#
#
# 2) **Predikering av et nytt punkt**: La oss si at du finner en ny iris blomst med 
# sepal length = 5.0cm, sepal width = 4.0 cm, petal length = 4.0 cm, og petal width = 2.0 cm. 
# Predikere typen av iris for denne blomsten.
#
# Det kan være nyttig å observere at X_test hadde formaten: \
# [[6.1, 2.8, 4.7, 1.2],  [5.7, 3.8, 1.7, 0.3], ..., [7.7, 2.6, 6.9, 2.3]]
#        

# %% [markdown]
#

# %% [markdown]
# # Regresjon med California housing data
#
#
#
# #### Bruk *2_CA_housing_regresjon.py*.
#
# 1) **Prøv et nevralt nettverk.** 
#
# Finne cellen der det står "Prøv med dine egne modelldefinisjoner her". Hva er mean average error (MAE) på testsettet når vi trener et nevralt nettverk?
#
# Du kan bruke en multi-layor perceptron regressor, som er en klassisk algoritme for nevralte nettverker, og spesifisere modellen sånn: \
# <code>model = MLPRegressor(hidden_layer_sizes=(30,))</code>
#
#
# 2) **Justering av hyperparametre (tuning)**. Som standard kan random forest modeller har bare et punkt fra treningssettet på en bladnode. En måte for å gjøre "smoothing" er å kreve at det er flere enn et punkt i hver bladnode (d.v.s. at modellen blir mindre tilpasset til treningssettet). Vi kan justere denne hyperparameteren med *min_samples_leaf* innstillingen, f.eks.:\
# <code>RandomForestRegressor(min_samples_leaf=2)</code>\
# \
# A) Bruk <code>GridSearchCV(...)</code> for å finne ut hvilken innstilling av *min_samples_leaf* parameteren til random forest fungerer best for våre data: 1, 2, eller 3. \
#  \
#  B) Hva er MAE ytelse i testsettet for den beste innstillingen du har funnet ovenfor? 
#  
#
# # Klassifisering med California housing data
#  
#  
#
# #### Bruk *3_CA_housing_klassifisering.py*.
#
# 1) **Prøv K-nærmeste nabo (K nearest neighbors, KNN).** 
#
# Finne cellen der det står "Prøv med dine egne modelldefinisjoner her". Hva er accuracy på testsettet med KNN? Hva er F1 score på testsettet?
#
# KNN er implementert i python sklearn sånn: \
# <code>model = KNeighborsClassifier()</code>
#
#
# 2) **(Valgfritt, litt mer involvert eksempel.) Bruk regresjons-løsningen for klassifisering.** 
#
#
#
# I den siste modullen har vi bygd en random forest modell som predikerer median_house_value fra X-variablene. Den modellen gir mye rikere informasjon enn hva vi trenger har, som er boligverdier i en grov klassifiseringsordning. Hva om vi brukte bare den modellen, og konverterte husverdiene til kategorier?\
#
# Gå tilbake til 2_CA_housing_regresjon.py, og finn implementasjon av random forest. Konvertere de predikerte numeriske husverdiene til kategorier. Her er konverteringsfunksjonen vi brukte:
#
# <code>def dollars_to_categ(house_vals):
#     house_categs= np.select(
#     [   
#         house_vals < thresh_lowmed,
#         house_vals.between(thresh_lowmed, thresh_medhigh, inclusive='both'), 
#         house_vals.between(thresh_medhigh, thresh_veryhigh, inclusive='neither'),
#         house_vals>=thresh_veryhigh   ], 
#     [   'Low', 
#         'Medium',
#         'High',
#         'Very pricy'   ])
#     return(house_categs)
# </code>
#
# A) Hva er accuracy på testsettet når du bruker denne metoden? Hva er F1 score? \
# \
# B) Er den beste modellen for regresjon nødvendigvis den beste for klassifisering? Hvorfor?

# %%





# %%




