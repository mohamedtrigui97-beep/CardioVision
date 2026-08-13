import pandas as pd
import os

INPUT  = "data/heart_disease_uci.csv"
OUTPUT = "data/heart_disease_clean.csv"

# ─────────────────────────────────────────────
#  Chargement
# ─────────────────────────────────────────────
if not os.path.exists(INPUT) or os.path.getsize(INPUT) == 0:
    print(f"ERREUR : '{INPUT}' est introuvable ou vide.")
    exit(1)

df = pd.read_csv(INPUT)

# ─────────────────────────────────────────────
#  AVANT nettoyage
# ─────────────────────────────────────────────
print("=" * 60)
print("  AVANT NETTOYAGE")
print("=" * 60)
print(f"  Lignes   : {df.shape[0]}")
print(f"  Colonnes : {df.shape[1]}")
missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]
if missing_before.empty:
    print("  Valeurs manquantes : aucune")
else:
    print("\n  Valeurs manquantes par colonne :")
    for col, n in missing_before.items():
        pct = n / len(df) * 100
        print(f"    {col:<20} {n:>4}  ({pct:.1f}%)")

# ─────────────────────────────────────────────
#  ÉTAPE 1 — Créer la colonne 'target'
# ─────────────────────────────────────────────
df["target"] = (df["num"] != 0).astype(int)
print(f"\n[1] Colonne 'target' créée  "
      f"(0 = sain: {(df['target']==0).sum()}, "
      f"1 = malade: {(df['target']==1).sum()})")

# ─────────────────────────────────────────────
#  ÉTAPE 2 — Suppression des colonnes
# ─────────────────────────────────────────────
cols_to_drop = [c for c in ["ca", "thal", "id", "num"] if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"[2] Colonnes supprimées : {cols_to_drop}")
print(f"    Colonnes restantes  : {list(df.columns)}")

# ─────────────────────────────────────────────
#  ÉTAPE 3 — Suppression des lignes trestbps == 0
# ─────────────────────────────────────────────
if "trestbps" in df.columns:
    n_before = len(df)
    df = df[df["trestbps"] != 0]
    n_removed = n_before - len(df)
    print(f"[3] Lignes supprimées (trestbps == 0) : {n_removed}")
else:
    print("[3] Colonne 'trestbps' absente, étape ignorée.")

# ─────────────────────────────────────────────
#  ÉTAPE 4 — Imputation des valeurs manquantes
# ─────────────────────────────────────────────
num_cols  = df.select_dtypes(include="number").columns.tolist()
cat_cols  = df.select_dtypes(exclude="number").columns.tolist()

imputed = []

for col in num_cols:
    n = df[col].isnull().sum()
    if n > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        imputed.append(f"    {col:<20} mediane={med:.2f}  ({n} valeurs)")

for col in cat_cols:
    n = df[col].isnull().sum()
    if n > 0:
        mod = df[col].mode()[0]
        df[col] = df[col].fillna(mod)
        imputed.append(f"    {col:<20} mode='{mod}'  ({n} valeurs)")

if imputed:
    print(f"[4] Imputation des valeurs manquantes :")
    for msg in imputed:
        print(msg)
else:
    print("[4] Aucune valeur manquante à imputer.")

# ─────────────────────────────────────────────
#  ÉTAPE 5 — Sauvegarde
# ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"[5] Fichier sauvegardé -> {OUTPUT}")

# ─────────────────────────────────────────────
#  APRÈS nettoyage
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  APRÈS NETTOYAGE")
print("=" * 60)
print(f"  Lignes   : {df.shape[0]}")
print(f"  Colonnes : {df.shape[1]}")
missing_after = df.isnull().sum().sum()
print(f"  Valeurs manquantes totales : {missing_after}")
print(f"  Distribution target :")
print(df["target"].value_counts().to_string())
print("\n" + "=" * 60)
print("  Nettoyage terminé avec succes !")
print("=" * 60)
