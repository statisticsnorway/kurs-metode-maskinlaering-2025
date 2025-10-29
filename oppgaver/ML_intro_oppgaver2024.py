# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Klassifisering av iris data

# %% [markdown]
# #### Bruk *1_min_forste_ML_program_iris.py*.

# %% [markdown]
# 1) **Se på trening / test split**: Kjør test/ training split med 75% i testdata (nå er det 25% i testdata). Hva er weighted average precision med den nye innstillingen?
#  

# %% [markdown]
#

# %% [markdown]
# 2) **Predikering av et nytt punkt**: La oss si at du finner en ny iris blomst med 
# sepal length = 5.0cm, sepal width = 4.0 cm, petal length = 4.0 cm, og petal width = 2.0 cm. 
# Predikere typen av iris for denne blomsten.
#
# Det kan være nyttig å observere at X_test hadde formaten: \
# [[6.1, 2.8, 4.7, 1.2],  [5.7, 3.8, 1.7, 0.3], ..., [7.7, 2.6, 6.9, 2.3]]
#        

# %%

# %% [markdown]
# # Regresjon med California housing data
#
#

# %% [markdown]
# ### 1) Gjennomfør en hel maskinlæring arbeidsflyt på et nytt datasett.
#
#

# %% [markdown]
# Åpne **4_CA_housing_regresjon_oppgave.py**  filen. Den er en malfil du kan fylle med kode for å besvare de punktene nedenfor.\
#  \
# Her skal du gjennomføre en hel ML arbeidsflyt med random forest på et nytt datasett, *CA_housing_waterfront_regr.csv*.
# Dette datasettet er et utdrag av *CA_housing_1990.csv* dataene som var brukt i de demo filene, og inneholder bolig der 'ocean_proximity' variabel tok en av følgende verdier: (NEAR OCEAN, NEAR BAY, ISLAND).

# %% [markdown]
# A) Lese inn *CA_housing_waterfront_regr.csv*.

# %% [markdown]
#

# %% [markdown]
# B) Inspisere data. Er alle variablene numeriske? Er det manglende verdier?

# %% [markdown]
#

# %% [markdown]
# C) Lage en matrise **X** som inneholder de inngangsvariablene, og en array **y** som er utfallsvariabelen.

# %% [markdown]
#

# %% [markdown]
# D) Sjekk beskrivende statistikk. Du kan bruke syntaksen <code>mydataframe.describe()</code>

# %% [markdown]
#

# %% [markdown]
# E) Dele dataene i trenings- og testsetter. Sjekk størrelsen på begge settene.

# %% [markdown]
#

# %% [markdown]
# F) Trene en random forest modell. Du kan bruke standardinnstillingene for hyperparametrene, <code>model=RandomForestRegressor()</code>.

# %% [markdown]
#

# %% [markdown]
# G) Lage prediksjoner i testsettet.

# %% [markdown]
#

# %% [markdown]
# H) Vurdere ytelsen. Hvordan er MAE i dette mer homogene datasettet (\~5000 bolig som er alle ved vannkanten), sammenlignet med det opprinnelige datasettet (\~20000 bolig) som var brukt i *2_CA_housing_regresjon_demo.py*?

# %% [markdown]
#

# %% [markdown]
# I) Gjør hyperparameter justering (tuning). \
# \
# Som standard kan random forest modeller har bare et punkt fra treningssettet som gir verdien til en bladnode. En måte for å gjøre "smoothing" er å kreve at det er flere enn et punkt som er tildelt hver bladnode (d.v.s. at modellen blir mindre tilpasset til treningssettet). Vi kan justere denne hyperparameteren med *min_samples_leaf* innstillingen, f.eks.:\
# <code>RandomForestRegressor(min_samples_leaf=2)</code>\
# \
# Bruk <code>GridSearchCV(...)</code> for å finne ut hvilken innstilling av *min_samples_leaf* parameteren til random forest fungerer best for våre data: 1, 2, eller 3.
#

# %% [markdown]
#

# %% [markdown]
# J) Hva er den beste innstillingen av *min_samples_leaf* parameteren?

# %% [markdown]
#

# %% [markdown]
# K) Lage prediksjoner i testsettet med den beste innstillingen.

# %% [markdown]
#

# %% [markdown]
# L) Vurdere MAE ytelsen av random forest med denne innstillingen. Gjør hyperparameter justering en stor forskjell her?

# %% [markdown]
#

# %%

# %% [markdown]
# Bra jobba! :-) Nå har du bygd din egen maskinlæring arbeidsflyt helt fra bunnen av!

# %%

# %%

# %% [markdown]
# ### 2) Prøv en annen algoritme, ridge regression.
#
# Ridge regression er en algoritmen som bruker en lineær modell akkurat som lineær regresjon, men kontrollerer mot overtilpasning ved å foretrekke modeller som har minst mulig størrelse i koeffisientene. Kan vi bruke den til å bygge en enkelt lineær modell som er like lett å tolke som lineær regresjon, men samtidig har bedre ytelse? 
#
# Gå til *2_CA_housing_regresjon_demo.py* og finne cellen der det står "Prøv med dine egne modelldefinisjoner her". Hva er mean average error (MAE) på testsettet når vi trener en ridge regresjon modell? Er det bedre enn ytelsen av enkel lineær regresjon?

# %% [markdown]
# Du kan bruke versjonen av ridge regresjon som har innbygget kryssvalidering, og spesifisere modellen sånn: \
# <code>model = RidgeCV()</code>
#

# %%

# %%

# %%

# %% [markdown]
# # Klassifisering med California housing data
#
#

# %% [markdown]
# ### 1) Gjennomfør en hel maskinlæring arbeidsflyt på et nytt datasett.
#
#

# %% [markdown]
# Åpne **5_CA_housing_klassifisering_oppgave.py**  filen. Den er en malfil du kan fylle med kode for å besvare de punktene nedenfor.\
#  \
# Her skal du gjennomføre en hel ML arbeidsflyt med support vector machine på et nytt datasett, *CA_housing_waterfront_classif.csv*.
# Dette datasettet er et utdrag av *CA_housing_1990.csv* dataene som var brukt i de demo filene, og inneholder bolig der 'ocean_proximity' variabel tok en av følgende verdier: (NEAR OCEAN, NEAR BAY, ISLAND).

# %% [markdown]
# A) Lese inn *CA_housing_waterfront_classif.csv*.

# %% [markdown]
#

# %% [markdown]
# B) Inspisere data. Er alle variablene numeriske? Er det manglende verdier?

# %% [markdown]
#

# %% [markdown]
# C) Lage en matrise **X** som inneholder de inngangsvariablene, og en array **y** som er utfallsvariabelen.

# %% [markdown]
#

# %% [markdown]
# D) Sjekk beskrivende statistikk. Du kan bruke syntaksen <code>mydataframe.describe()</code>

# %% [markdown]
#

# %% [markdown]
# E) Dele data i trenings- og testsetter. Sjekk størrelsen på begge settene.

# %% [markdown]
#

# %% [markdown]
# F) Trene en support vector machine modellen. \
# \
# Du kan bruke standardinnstillingene for hyperparametrene, og bruke <code>model=SVC()</code>  ('SVC' betyr support vector classifier).

# %% [markdown]
#

# %% [markdown]
# G) Lage prediksjoner i testsettet.

# %% [markdown]
#

# %% [markdown]
# H) Vurdere ytelsen. Hvordan er nøyaktigheten i dette mer homogene datasettet (\~5000 bolig som er alle ved vannkanten), sammenlignet med det opprinnelige datasettet (\~20000 bolig) som var brukt i *2_CA_housing_klassifisering_demo.py*? Hva med F1 score verdier?

# %% [markdown]
#

# %% [markdown]
# I) Skriv ut en klassifiseringsrapport. For hvilken kategori av boligverdier (High, Low, Medium, Very pricy) identifiserer modellen den høyeste prosentandelen av tilfellene? For hvilken kategori har modellen den høyeste prosentandelen korrekte av tilfeller den har identifisert?

# %% [markdown]
#

# %%

# %%

# %% [markdown]
# ### 2) Prøv K-nærmeste nabo (K nearest neighbors, KNN).
#
# Gå til *3_CA_housing_regresjon_demo.py* og finne cellen der det står "Prøv med dine egne modelldefinisjoner her". Hva er accuracy på testsettet med KNN? Hva er F1 score på testsettet?

# %% [markdown]
# KNN er implementert i python sklearn sånn: \
# <code>model = KNeighborsClassifier()</code>
#

# %%

# %%

# %% [markdown]
# ### 3) (Valgfritt, litt mer involvert eksempel.) Bruk regresjons-løsningen for klassifisering.
#
#

# %% [markdown]
# I den siste modullen har vi bygd en random forest modell som predikerer median_house_value fra X-variablene. Den modellen gir mye rikere informasjon enn hva vi trenger har, som er boligverdier i en grov klassifiseringsordning. Hva om vi brukte bare den modellen, og konverterte husverdiene til kategorier?\
# \
# Gå tilbake til *2_CA_housing_regresjon_demo.py*, og finn implementasjonen av random forest. Konvertere de predikerte numeriske husverdiene til kategorier. Her er konverteringsfunksjonen vi brukte:

# %% [markdown]
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

# %% [markdown]
# A) Hva er accuracy på testsettet når du bruker denne metoden? Hva er F1 score? \

# %%

# %% [markdown]
# B) Er den beste modellen for regresjon nødvendigvis den beste for klassifisering? Hvorfor?

# %%
