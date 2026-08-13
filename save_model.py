import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_save_model():
    data_path = "data/heart_disease_clean.csv"
    model_output_path = "data/heart_disease_pipeline.joblib"

    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier {data_path} n'existe pas.")
        return

    print("Chargement des données nettoyées...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    print(f"Features numériques ({len(num_cols)}) : {num_cols}")
    print(f"Features catégorielles ({len(cat_cols)}) : {cat_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])

    print("Entraînement du modèle Pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy sur test : {acc:.4f} ({acc*100:.1f}%)")
    print(f"AUC-ROC sur test  : {auc:.4f}")

    # Ré-entraînement sur l'ensemble complet des données
    print("Ré-entraînement sur l'ensemble complet des données...")
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"[OK] Modele sauvegarde avec succes dans : {model_output_path}")

if __name__ == "__main__":
    train_and_save_model()

