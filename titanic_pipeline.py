"""Pipeline de classification supervisée : Titanic (survie des passagers).

Ce script reprend le code du notebook 03-Preprocessing_with_sklearn.ipynb,
étape par étape, sans fonctions.

Objectif : prédire si un passager a survécu (colonne "Survived") à partir
de ses informations (classe, sexe, âge, tarif, port d'embarquement...).
    - Y (la cible) : la colonne "Survived"
    - X (les variables explicatives) : toutes les autres colonnes utiles

Usage :
    python titanic_pipeline.py
"""

# =============================================================================
# ÉTAPE 0 : IMPORTS
# =============================================================================
# On importe toutes les bibliothèques dont on aura besoin :
#   - pandas / numpy : manipulation des données
#   - train_test_split : découpage en train set / test set
#   - Pipeline, ColumnTransformer : organisation du preprocessing
#   - SimpleImputer : remplacement des valeurs manquantes
#   - OneHotEncoder, StandardScaler, LabelEncoder : transformations des colonnes
#   - LogisticRegression : le modèle
#   - accuracy_score : la métrique d'évaluation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning
)  # to avoid deprecation warnings


# =============================================================================
# ÉTAPE 1 : CHARGEMENT DU DATASET
# =============================================================================
# On lit le fichier CSV et on le stocke dans un DataFrame pandas.
# Le chemin "src/titanic.csv" est relatif : lance le script depuis le dossier
# qui contient le sous-dossier "src".

print("Loading dataset...")
dataset = pd.read_csv("src/titanic.csv")
print("...Done.")
print()


# =============================================================================
# ÉTAPE 2 : EXPLORATION RAPIDE
# =============================================================================
# On regarde le nombre de lignes, les premières lignes, des statistiques
# de base et le pourcentage de valeurs manquantes par colonne.
# Ces informations servent à décider quel preprocessing faire ensuite.
#
# Remarque : dans le notebook, on utilisait display(). Cette fonction
# n'existe que dans Jupyter, donc ici elle est remplacée par print().

print("Number of rows : {}".format(dataset.shape[0]))
print()

print("Display of dataset: ")
print(dataset.head())
print()

print("Basics statistics: ")
data_desc = dataset.describe(include="all")
print(data_desc)
print()

print("Percentage of missing values: ")
print(100 * dataset.isnull().sum() / dataset.shape[0])
print()


# =============================================================================
# ÉTAPE 3 : PREPROCESSING PANDAS - SUPPRESSION DES COLONNES INUTILES
# =============================================================================
# Certaines colonnes n'apportent pas d'information utile pour la prédiction :
#   - PassengerId : un simple identifiant
#   - Ticket : un numéro de billet, presque unique pour chaque passager
#   - Name : le nom du passager
# axis=1 signifie qu'on supprime des colonnes (axis=0 supprimerait des lignes).

useless_cols = ["PassengerId", "Ticket", "Name"]

print("Dropping useless columns...")
dataset = dataset.drop(
    useless_cols, axis=1
)  # axis = 1 indicates that we are dropping along the column axis
print("...Done.")
print(dataset.head())
print()


# =============================================================================
# ÉTAPE 4 : PREPROCESSING PANDAS - SUPPRESSION DES OUTLIERS (MASQUES)
# =============================================================================
# On supprime les lignes contenant des valeurs aberrantes, avec des masques :
#   - Age : on garde les âges positifs OU manquants (les valeurs manquantes
#     seront remplacées plus tard par le SimpleImputer)
#   - Fare : on garde les tarifs inférieurs à moyenne + 2 écarts-types

print("Dropping outliers in Age...")
to_keep = (dataset["Age"] > 0) | (
    dataset["Age"].isnull()
)  # We want keeping positives values or missings
dataset = dataset.loc[to_keep, :]
print("Done. Number of lines remaining : ", dataset.shape[0])
print()

print("Dropping outliers in Fare...")
to_keep = dataset["Fare"] < dataset["Fare"].mean() + 2 * dataset["Fare"].std()
dataset = dataset.loc[to_keep, :]
print("Done. Number of lines remaining : ", dataset.shape[0])
print()


# =============================================================================
# ÉTAPE 5 : SÉPARATION DE LA CIBLE (Y) ET DES VARIABLES EXPLICATIVES (X)
# =============================================================================
# Y contient uniquement la colonne à prédire ("Survived").
# X contient toutes les autres colonnes.

target_S = "Survived"

print("Separating labels from features...")
Y = dataset.loc[:, target_S]
X = dataset.drop(target_S, axis=1)  # All columns are kept, except the target
print("...Done.")
print(Y.head())
print()
print(X.head())
print()


# =============================================================================
# ÉTAPE 6 : DÉCOUPAGE TRAIN SET / TEST SET
# =============================================================================
# On met de côté 20 % des données (le test set) : le modèle ne les verra
# jamais pendant l'entraînement. Elles servent à estimer ses performances
# sur des données nouvelles.
# Règle : toujours faire ce découpage AVANT les transformations scikit-learn.
#   - test_size=0.2 : 20 % des lignes vont dans le test set
#   - random_state=0 : le découpage "aléatoire" est le même à chaque exécution

print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("...Done.")
print()


# =============================================================================
# ÉTAPE 7 : PREPROCESSING SCIKIT-LEARN - PIPELINE DES COLONNES NUMÉRIQUES
# =============================================================================
# Pour les colonnes numériques, deux étapes successives :
#   1. imputer : remplace les valeurs manquantes par la médiane de la colonne
#   2. scaler : standardise la colonne (moyenne 0, écart-type 1)

numeric_features = ["Age", "Parch", "Fare"]  # Names of numeric columns in X_train/X_test
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),  # missing values will be replaced by columns' median
        ("scaler", StandardScaler()),
    ]
)


# =============================================================================
# ÉTAPE 8 : PREPROCESSING SCIKIT-LEARN - PIPELINE DES COLONNES CATÉGORIELLES
# =============================================================================
# Pour les colonnes catégorielles, deux étapes successives :
#   1. imputer : remplace les valeurs manquantes par la valeur la plus fréquente
#   2. encoder : one-hot encoding (une colonne 0/1 par catégorie).
#      drop="first" supprime la première colonne créée, pour éviter
#      des colonnes redondantes (corrélées entre elles).

categorical_features = ["Pclass", "Embarked", "Sex"]  # Names of categorical columns in X_train/X_test
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),  # missing values will be replaced by most frequent value
        (
            "encoder",
            OneHotEncoder(drop="first"),
        ),  # first column will be dropped to avoid creating correlations between features
    ]
)


# =============================================================================
# ÉTAPE 9 : PREPROCESSING SCIKIT-LEARN - COLUMNTRANSFORMER
# =============================================================================
# Le ColumnTransformer applique chaque pipeline aux bonnes colonnes :
#   - "num" : numeric_transformer sur les colonnes numériques
#   - "cat" : categorical_transformer sur les colonnes catégorielles
# Les colonnes qui ne sont listées nulle part (SibSp, Cabin) sont supprimées.

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# =============================================================================
# ÉTAPE 10 : APPLICATION DU PREPROCESSING SUR LE TRAIN ET LE TEST
# =============================================================================
#   - Sur le train set : fit_transform() => le préprocesseur APPREND les
#     paramètres (médianes, moyennes, catégories...) puis les applique.
#   - Sur le test set : transform() seulement => on réutilise ce qui a été
#     appris sur le train. Refaire un fit sur le test créerait une "fuite
#     de données" (data leakage) qui fausserait les résultats.
#
# ATTENTION : après cette étape, X_train et X_test ne sont plus des
# DataFrames pandas mais des tableaux numpy. On ne peut donc plus utiliser
# .head() : on affiche les 5 premières lignes avec X_train[0:5].

print("Performing preprocessings on train set...")
print(X_train.head())
X_train = preprocessor.fit_transform(X_train)
print("...Done.")
print(
    X_train[0:5]
)  # MUST use this syntax because X_train is a numpy array and not a pandas DataFrame anymore
print()

print("Performing preprocessings on test set...")
print(X_test.head())
X_test = preprocessor.transform(
    X_test
)  # Don't fit again !! The test set is used for validating decisions
# we made based on the training set, therefore we can only apply transformations that were parametered using the training set.
# Otherwise this creates what is called a leak from the test set which will introduce a bias in all your results.
print("...Done.")
print(
    X_test[0:5, :]
)  # MUST use this syntax because X_test is a numpy array and not a pandas DataFrame anymore
print()

# Dans le notebook, la cellule suivante provoquait volontairement une erreur.
# Elle est laissée en commentaire pour ne pas arrêter le script :
# X_train.head()  # AttributeError: 'numpy.ndarray' object has no attribute 'head'

print("Type of X_train after preprocessing:", type(X_train))
print()


# =============================================================================
# ÉTAPE 11 : ENCODAGE DE LA CIBLE Y
# =============================================================================
# Le LabelEncoder transforme les valeurs de la cible en nombres 0, 1, 2...
# Ici, "Survived" contient déjà 0 et 1, donc le résultat ne change pas,
# mais cette étape est indispensable quand la cible est du texte
# (par exemple "yes" / "no").
# Même règle que pour X : fit_transform sur le train, transform sur le test.

labelencoder = LabelEncoder()

print("Encoding labels on train set...")
print(Y_train.head())
print()
Y_train = labelencoder.fit_transform(Y_train)
print("...Done.")
print(Y_train[0:5])
print()

print("Encoding labels on test set...")
print(Y_test.head())
print()
Y_test = labelencoder.transform(Y_test)  # Don't fit again !!
print("...Done.")
print(Y_test[0:5])
print()


# =============================================================================
# ÉTAPE 12 : ENTRAÎNEMENT DU MODÈLE
# =============================================================================
# On crée une régression logistique et on l'entraîne avec fit().
# L'entraînement se fait TOUJOURS sur le train set uniquement.

model = LogisticRegression()

print("Training model...")
model.fit(X_train, Y_train)  # Training is always done on train set !!
print("...Done.")
print()


# =============================================================================
# ÉTAPE 13 : PRÉDICTIONS
# =============================================================================
# predict() renvoie la classe prédite (0 ou 1) pour chaque ligne.
# On prédit sur le train ET sur le test pour pouvoir comparer les scores.

print("Predictions on training set...")
Y_train_pred = model.predict(X_train)
print("...Done.")
print(Y_train_pred[0:5])
print()

print("Predictions on test set...")
Y_test_pred = model.predict(X_test)
print("...Done.")
print(Y_test_pred[0:5])
print()


# =============================================================================
# ÉTAPE 14 : ÉVALUATION DES PERFORMANCES
# =============================================================================
# L'accuracy est la proportion de bonnes prédictions.
# accuracy_score(vraies_valeurs, prédictions) : les vraies valeurs en premier.
#
# Comment lire les résultats :
#   - score train >> score test : le modèle apprend "par cœur" (overfitting)
#   - scores proches : le modèle généralise bien

print(
    "Accuracy on training set : ", accuracy_score(Y_train, Y_train_pred)
)  # Always pass true label first, and predictions in second position
print("Accuracy on test set : ", accuracy_score(Y_test, Y_test_pred))
print()


# =============================================================================
# ÉTAPE 15 : MÉTHODE ALTERNATIVE - model.score()
# =============================================================================
# Chaque modèle scikit-learn a une méthode score() qui calcule directement
# le score, sans appeler predict() avant. Pour un classifieur comme la
# LogisticRegression, ce score est l'accuracy.
# Attention à l'ordre : ici les variables X en premier, puis les vraies valeurs.

print(
    "Accuracy on training set : ", model.score(X_train, Y_train)
)  # Here, the features must be passed first, and then the true label
print("Accuracy on test set : ", model.score(X_test, Y_test))
