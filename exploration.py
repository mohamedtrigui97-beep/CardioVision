import pandas as pd
import os

filepath = "data/heart_disease_uci.csv"

# Vérification du fichier
if not os.path.exists(filepath):
    print(f"ERREUR : Fichier introuvable -> {filepath}")
else:
    size = os.path.getsize(filepath)
    print(f"Fichier trouvé : {filepath}  ({size} octets)\n")

    if size == 0:
        print("ERREUR : Le fichier est vide (0 octets). Veuillez vérifier le fichier.")
    else:
        df = pd.read_csv(filepath)

        print("=" * 55)
        print("  FORME DU DATASET")
        print("=" * 55)
        print(f"  Lignes   : {df.shape[0]}")
        print(f"  Colonnes : {df.shape[1]}")

        print("\n" + "=" * 55)
        print("  COLONNES ET TYPES")
        print("=" * 55)
        for col, dtype in df.dtypes.items():
            print(f"  {col:<25} {str(dtype)}")

        print("\n" + "=" * 55)
        print("  VALEURS MANQUANTES PAR COLONNE")
        print("=" * 55)
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({"Manquantes": missing, "Pourcentage (%)": pct})
        report = report[report["Manquantes"] > 0].sort_values("Manquantes", ascending=False)
        if report.empty:
            print("  Aucune valeur manquante détectée.")
        else:
            print(report.to_string())

        print("\n" + "=" * 55)
        print("  APERCU (5 premières lignes)")
        print("=" * 55)
        print(df.head().to_string())
