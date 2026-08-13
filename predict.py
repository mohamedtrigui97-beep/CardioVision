import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ─────────────────────────────────────────────
#  Couleurs terminal
# ─────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
GRAY   = "\033[90m"

def titre(texte):
    print(f"\n{BOLD}{CYAN}{'='*55}{RESET}")
    print(f"{BOLD}{CYAN}  {texte}{RESET}")
    print(f"{BOLD}{CYAN}{'='*55}{RESET}")

def question(texte, options=None):
    print(f"\n{YELLOW}{BOLD}{texte}{RESET}")
    if options:
        for i, opt in enumerate(options, 1):
            print(f"  {GRAY}{i}.{RESET} {opt}")
    return input(f"  {BLUE}> {RESET}").strip()

# ─────────────────────────────────────────────
#  1. Entraînement du modèle
# ─────────────────────────────────────────────
titre("Chargement et entraînement du modèle...")

df = pd.read_csv("data/heart_disease_clean.csv")
X  = df.drop(columns=["target"])
y  = df["target"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
])

pipeline.fit(X, y)
print(f"  {GREEN}✓ Modèle entraîné sur {len(df)} patients !{RESET}")

# ─────────────────────────────────────────────
#  2. Saisie des paramètres du patient
# ─────────────────────────────────────────────
titre("Saisie des paramètres du patient")
print(f"  {GRAY}Appuyez sur Entrée après chaque valeur.{RESET}")

def saisir_nombre(texte, min_val, max_val, type_=float):
    while True:
        try:
            val = type_(question(texte))
            if min_val <= val <= max_val:
                return val
            print(f"  {RED}Valeur hors limites ({min_val} - {max_val}). Réessayez.{RESET}")
        except ValueError:
            print(f"  {RED}Valeur invalide. Entrez un nombre.{RESET}")

def saisir_choix(texte, options_map):
    opts = list(options_map.keys())
    while True:
        rep = question(texte, opts)
        try:
            idx = int(rep) - 1
            if 0 <= idx < len(opts):
                return options_map[opts[idx]]
        except ValueError:
            pass
        # Accepter aussi la valeur directe
        if rep in options_map.values():
            return rep
        print(f"  {RED}Choix invalide. Entrez un numéro entre 1 et {len(opts)}.{RESET}")

# ── Paramètres
age      = saisir_nombre("Âge du patient (années) [1-120]", 1, 120, int)

sex = saisir_choix(
    "Sexe du patient :",
    {"Homme": "Male", "Femme": "Female"}
)

cp = saisir_choix(
    "Type de douleur thoracique (cp) :",
    {
        "Angine typique (angine d'effort classique)":      "typical angina",
        "Angine atypique (douleur non caractéristique)":   "atypical angina",
        "Douleur non-angineuse (autre douleur thoracique)":"non-anginal",
        "Asymptomatique (aucune douleur)":                 "asymptomatic",
    }
)

trestbps = saisir_nombre("Pression artérielle au repos (mm Hg) [80-250]", 80, 250, float)
chol     = saisir_nombre("Taux de cholestérol (mg/dl) [100-600]", 100, 600, float)

fbs = saisir_choix(
    "Glycémie à jeun > 120 mg/dl ?",
    {"Oui (> 120 mg/dl)": "True", "Non (≤ 120 mg/dl)": "False"}
)

restecg = saisir_choix(
    "Résultat ECG au repos :",
    {
        "Normal":                        "normal",
        "Anomalie onde ST-T":            "st-t abnormality",
        "Hypertrophie ventriculaire gauche": "lv hypertrophy",
    }
)

thalch = saisir_nombre("Fréquence cardiaque maximale atteinte [60-250]", 60, 250, float)

exang = saisir_choix(
    "Angine induite par l'exercice ?",
    {"Oui": "True", "Non": "False"}
)

oldpeak = saisir_nombre("Dépression ST induite par l'exercice [0.0-10.0]", 0.0, 10.0, float)

slope = saisir_choix(
    "Pente du segment ST en exercice :",
    {
        "Montante (upsloping)":   "upsloping",
        "Plate (flat)":           "flat",
        "Descendante (downsloping)": "downsloping",
    }
)

# Dataset = source clinique (Cleveland par défaut pour un nouveau patient)
dataset = "Cleveland"

# ─────────────────────────────────────────────
#  3. Prédiction
# ─────────────────────────────────────────────
patient = pd.DataFrame([{
    "age":      age,
    "sex":      sex,
    "dataset":  dataset,
    "cp":       cp,
    "trestbps": trestbps,
    "chol":     chol,
    "fbs":      fbs,
    "restecg":  restecg,
    "thalch":   thalch,
    "exang":    exang,
    "oldpeak":  oldpeak,
    "slope":    slope,
}])

prediction  = pipeline.predict(patient)[0]
probabilite = pipeline.predict_proba(patient)[0]
prob_malade = probabilite[1] * 100
prob_sain   = probabilite[0] * 100

# ─────────────────────────────────────────────
#  4. Affichage du résultat
# ─────────────────────────────────────────────
titre("RÉSULTAT DU DIAGNOSTIC")

print(f"\n  {'─'*50}")
print(f"  {BOLD}Récapitulatif du patient :{RESET}")
print(f"  {'─'*50}")
print(f"  Âge              : {age} ans")
print(f"  Sexe             : {sex}")
print(f"  Douleur (cp)     : {cp}")
print(f"  Pression artér.  : {trestbps} mm Hg")
print(f"  Cholestérol      : {chol} mg/dl")
print(f"  Glycémie élevée  : {fbs}")
print(f"  ECG              : {restecg}")
print(f"  FC max           : {thalch} bpm")
print(f"  Angine exercice  : {exang}")
print(f"  Dépression ST    : {oldpeak}")
print(f"  Pente ST         : {slope}")
print(f"  {'─'*50}")

if prediction == 1:
    print(f"\n  {RED}{BOLD}⚠  DIAGNOSTIC : MALADIE CARDIAQUE PROBABLE{RESET}")
    print(f"  {RED}Probabilité d'être malade  : {prob_malade:.1f}%{RESET}")
    print(f"  {GREEN}Probabilité d'être sain    : {prob_sain:.1f}%{RESET}")
    print(f"\n  {YELLOW}⚕  Recommandation : Consulter un cardiologue.{RESET}")
else:
    print(f"\n  {GREEN}{BOLD}✓  DIAGNOSTIC : PAS DE MALADIE CARDIAQUE DÉTECTÉE{RESET}")
    print(f"  {GREEN}Probabilité d'être sain    : {prob_sain:.1f}%{RESET}")
    print(f"  {RED}Probabilité d'être malade  : {prob_malade:.1f}%{RESET}")
    print(f"\n  {YELLOW}⚕  Recommandation : Continuer les contrôles réguliers.{RESET}")

# ── Barre de probabilité visuelle
print(f"\n  Risque  [", end="")
nb_rouge = int(prob_malade / 5)
nb_vert  = 20 - nb_rouge
print(f"{RED}{'█' * nb_rouge}{GREEN}{'█' * nb_vert}{RESET}]  {prob_malade:.1f}% risque")

print(f"\n  {GRAY}⚠ Avertissement : Ce diagnostic est une aide à la décision,")
print(f"  il ne remplace pas l'avis d'un professionnel de santé.{RESET}")
print(f"\n{CYAN}{'='*55}{RESET}\n")
