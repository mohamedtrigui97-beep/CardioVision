# 🫀 CardioCheck AI — Document Explicatif du Projet

Ce document présente une explication détaillée et structurée de l'ensemble des fichiers, scripts, données et artefacts composant le projet **CardioCheck AI** (Système de Prédiction et Détection des Maladies Cardiaques par Apprentissage Automatique).

---

## 📁 Vue d'Ensemble de l'Architecture du Projet

```text
Nouveau dossier/
│
├── 📜 README.md                         --> Document explicatif complet du projet (ce fichier)
│
├── 📊 Scripts Python (Pipeline Machine Learning & Applications)
│   ├── exploration.py                  --> Analyse exploratoire du dataset brut
│   ├── cleaning.py                     --> Nettoyage des données et imputation
│   ├── logistic_regression.py          --> Entraînement, évaluation & visualisations du modèle
│   ├── save_model.py                   --> Persistance du modèle final sous format Joblib
│   ├── predict.py                      --> Interface CLI interactive de prédiction pour terminal
│   └── app.py                          --> Application Web interactive Streamlit (Interface Médicale)
│
└── 📂 data/                             --> Dossier contenant les données, modèle et visualisations
    ├── heart_disease_uci.csv           --> Dataset brut initial (UCI Heart Disease)
    ├── heart_disease_clean.csv         --> Dataset nettoyé et imputé prêt pour le ML
    ├── heart_disease_pipeline.joblib   --> Modèle entraîné et son Pipeline Scikit-Learn sérialisé
    └── logistic_regression_results.png --> Graphique haute résolution des performances du modèle
```

---

## 📄 Description Détaillée des Fichiers

### 1. 🔍 `exploration.py` (Analyse Exploratoire Initiale)
* **Rôle** : Inspecte la structure et la qualité des données brutes contenues dans `data/heart_disease_uci.csv`.
* **Fonctionnalités Principales** :
  * Vérifie l'existence et la taille du fichier.
  * Affiche les dimensions du dataset (nombre de lignes et de colonnes).
  * Liste le nom des variables et leurs types de données (`dtypes`).
  * Calcule et affiche le nombre ainsi que le pourcentage de valeurs manquantes par colonne.
  * Affiche un aperçu des 5 premières lignes du jeu de données.

---

### 2. 🧹 `cleaning.py` (Pipeline de Nettoyage & Préparation)
* **Rôle** : Transforme le jeu de données brut en un dataset propre et prêt pour l'entraînement.
* **Étapes de Traitement** :
  1. **Création de la variable cible (`target`)** : Convertit la colonne `num` (gravité de 0 à 4) en classification binaire ($0 = \text{Sain}$, $1 = \text{Malade}$).
  2. **Suppression des colonnes inutiles** : Retire `id`, `num`, `ca`, et `thal` (trop de valeurs manquantes ou redondantes).
  3. **Filtrage des valeurs aberrantes** : Supprime les entrées physiologiquement impossibles (ex: `trestbps == 0`).
  4. **Imputation des valeurs manquantes** :
     * Imputation par la **médiane** pour les variables numériques (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`).
     * Imputation par le **mode** (valeur la plus fréquente) pour les variables catégorielles (`sex`, `dataset`, `cp`, `fbs`, `restecg`, `exang`, `slope`).
  5. **Sauvegarde** : Exporte le résultat vers `data/heart_disease_clean.csv`.

---

### 3. 📈 `logistic_regression.py` (Entraînement, Évaluation & Visualisations)
* **Rôle** : Construit, évalue et analyse le modèle de Régression Logistique avec validation croisée et visualisations complètes.
* **Fonctionnalités Principales** :
  * Construit un `Pipeline` Scikit-Learn combinant un `ColumnTransformer` (`StandardScaler` pour les numériques, `OneHotEncoder` pour les catégorielles) et `LogisticRegression`.
  * Sépare les données en ensembles d'entraînement (80%) et de test (20%) avec stratification.
  * Évalue le modèle via **Accuracy**, **AUC-ROC**, **Précision**, **Rappel**, **F1-Score**, et **Validation Croisée (5-fold Cross-Validation)**.
  * Calcule le coefficient d'importance des facteurs de risque.
  * Génère un tableau de bord graphique de 6 panneaux sauvegardé dans `data/logistic_regression_results.png` :
    1. Matrice de Confusion
    2. Courbe ROC (AUC)
    3. Top 12 des Coefficients (Facteurs de risque)
    4. Distribution des probabilités prédites
    5. Scores de Validation Croisée (5 Folds)
    6. Récapitulatif visuel des métriques clés

---

### 4. 💾 `save_model.py` (Sauvegarde du Modèle Exportable)
* **Rôle** : Automatise l'entraînement global du modèle sur 100% des données propres et exporte le pipeline complet.
* **Fonctionnalités Principales** :
  * Ré-entraîne le `Pipeline` Scikit-Learn (Prétraitement + Modèle).
  * Sérialise et sauvegarde le pipeline sous `data/heart_disease_pipeline.joblib` via `joblib`.
  * Permet la réutilisation immédiate du modèle dans les applications CLI ou Web sans ré-entraînement.

---

### 5. 💻 `predict.py` (Interface de Prédiction en Ligne de Commande - CLI)
* **Rôle** : Offre un outil interactif dans le terminal permettant d'évaluer le risque d'un nouveau patient en temps réel.
* **Fonctionnalités Principales** :
  * Pose des questions guidées et sécurisées avec validation des types et bornes numériques (ex: Âge entre 1 et 120, tension entre 80 et 250 mm Hg).
  * Supporte le choix parmi des options claires pour le sexe, type de douleur thoracique, ECG, etc.
  * Calcule la probabilité exacte d'être sain ou malade.
  * Affiche un rapport visuel en terminal avec couleurs ANSI, barre de niveau de risque et conseils médicaux.

---

### 6. 🌐 `app.py` (Application Web Interactive Streamlit)
* **Rôle** : Interface graphique utilisateur (GUI) moderne et intuitive destinée aux praticiens et utilisateurs.
* **Fonctionnalités Principales** :
  * **Design Premium** : Thème sombre avec cartes en Glassmorphism, animations et typographie soignée.
  * **Formulaire Médical Complet** (Barre latérale) :
    * Informations générales (Âge, Sexe, Origine).
    * Signes vitaux (Pression artérielle, Cholestérol, Glycémie).
    * Symptômes & Examens (Douleur thoracique, ECG, Fréquence cardiaque, Dépression ST, Pente ST, Angine).
  * **Résultat du Diagnostic** :
    * Badges visuels de statut (Risque Élevé vs Normal).
    * Jauge de probabilité avec recommandations dynamiques.
    * Diagramme circulaire Matplotlib des probabilités.
    * Graphique des 6 principaux facteurs de risque personnels.
  * **Exportation de Rapport** : Bouton de téléchargement direct d'un rapport médical officiel au format `.txt`.

---

## 📂 Contenu du Dossier `data/`

| Fichier / Image | Description |
| :--- | :--- |
| **`heart_disease_uci.csv`** | Fichier CSV d'origine contenant 920 instances et 16 variables issues de 4 centres médicaux. |
| **`heart_disease_clean.csv`** | Fichier CSV nettoyé sans données manquantes, prêt à être consommé par les pipelines ML. |
| **`heart_disease_pipeline.joblib`** | Fichier binaire contenant l'objet Scikit-Learn Pipeline (`ColumnTransformer` + `LogisticRegression`) entraîné. |
| **`logistic_regression_results.png`** | Image haute définition (150 DPI) synthétisant les résultats de modélisation et performances du modèle. |

---

## 🚀 Comment Exécuter le Projet

### 1. Préréquis et Installation des Dépendances
```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

### 2. Exécution des Étapes (Workflow Machine Learning)
```bash
# Step 1: Analyser les données
python exploration.py

# Step 2: Nettoyer et imputer les données
python cleaning.py

# Step 3: Entraîner et évaluer le modèle (génère le graphique de résultats)
python logistic_regression.py

# Step 4: Exporter le modèle sérialisé (.joblib)
python save_model.py
```

### 3. Lancement des Interfaces de Prédiction

* **En Ligne de Commande (CLI)** :
  ```bash
  python predict.py
  ```

* **Application Web Interactive (Streamlit)** :
  ```bash
  streamlit run app.py
  ```

---
*Document généré automatiquement pour le projet **CardioCheck AI**.*
