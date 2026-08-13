# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration de la page Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCheck AI — Détection des Maladies Cardiaques",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Style CSS personnalisé (Thème sombre moderne & Glassmorphism)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Styles généraux */
    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
    }
    
    /* En-tête principal */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }

    /* Cartes de résultat */
    .card-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        color: #f87171;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.15);
    }
    .card-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(21, 128, 61, 0.1) 100%);
        border: 1.5px solid #22c55e;
        border-radius: 16px;
        padding: 24px;
        color: #4ade80;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.15);
    }

    .stat-badge {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 14px;
        color: #cbd5e1;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
    }

    /* Modificateurs pour widgets */
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Chargement ou entraînement automatique du modèle
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "data/heart_disease_pipeline.joblib"
DATA_PATH = "data/heart_disease_clean.csv"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            return pipeline
        except Exception as e:
            st.warning(f"Impossible de charger le modèle existant ({e}). Réentraînement...")
    
    # Auto-entraînement si le fichier n'existe pas encore
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=["target"])
        y = df["target"]
        
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ])
        
        pipeline.fit(X, y)
        os.makedirs("data", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline
    else:
        st.error(f"Erreur critique : Fichier de données {DATA_PATH} introuvable.")
        return None

pipeline = load_or_train_model()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Interface utilisateur & Sidebar
# ─────────────────────────────────────────────────────────────────────────────
# En-tête principal
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🫀 CardioCheck AI</h1>
        <div class="header-subtitle">
            Système expert d'aide au diagnostic des maladies cardiaques basé sur la Régression Logistique.
        </div>
        <div style="margin-top: 15px;">
            <span class="stat-badge">Modèle : Régression Logistique</span>
            <span class="stat-badge">Accuracy : ~84.8%</span>
            <span class="stat-badge">AUC-ROC : 0.912</span>
            <span class="stat-badge">Source : Dataset UCI Heart Disease</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.stop()

# Formulaire d'entrée dans le panneau latéral (Sidebar)
st.sidebar.header("📋 Paramètres du Patient")
st.sidebar.markdown("Renseignez les données médicales du patient pour évaluer le risque cardiaque.")

with st.sidebar.form(key="patient_form"):
    st.subheader("1. Informations Personnelles")
    age = st.number_input("Âge (ans)", min_value=1, max_value=120, value=55, step=1)
    sex = st.selectbox("Sexe", options=["Male", "Female"], format_func=lambda x: "Homme (Male)" if x == "Male" else "Femme (Female)")
    dataset = "Cleveland"  # valeur implicite pour garder la compatibilité avec le modèle

    st.subheader("2. Douleur & Symptômes")
    cp_options = {
        "typical angina": "Angine typique (douleur caractéristique d'effort)",
        "atypical angina": "Angine atypique (douleur non caractéristique)",
        "non-anginal": "Douleur non-angineuse (autre origine)",
        "asymptomatic": "Asymptomatique (aucune douleur)"
    }
    cp = st.selectbox("Type de douleur thoracique (cp)", options=list(cp_options.keys()), format_func=lambda x: cp_options[x], index=3)
    
    exang_options = {"True": "Oui", "False": "Non"}
    exang = st.selectbox("Angine induite par l'exercice ?", options=["False", "True"], format_func=lambda x: exang_options[x], index=0)

    st.subheader("3. Mesures Physiologiques")
    trestbps = st.number_input("Pression artérielle au repos (mm Hg)", min_value=70, max_value=250, value=130, step=1)
    chol = st.number_input("Taux de cholestérol sérique (mg/dl)", min_value=80, max_value=600, value=240, step=1)
    fbs = st.selectbox("Glycémie à jeun > 120 mg/dl ?", options=["False", "True"], format_func=lambda x: "Oui (> 120 mg/dl)" if x == "True" else "Non (≤ 120 mg/dl)", index=0)

    st.subheader("4. Résultats ECG & Épreuve d'effort")
    restecg_options = {
        "normal": "Normal",
        "st-t abnormality": "Anomalie de l'onde ST-T",
        "lv hypertrophy": "Hypertrophie ventriculaire gauche"
    }
    restecg = st.selectbox("Résultat ECG au repos", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x], index=0)
    
    thalch = st.number_input("Fréquence cardiaque max atteinte (bpm)", min_value=60, max_value=230, value=150, step=1)
    oldpeak = st.number_input("Dépression ST induite par l'effort", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    slope_options = {
        "upsloping": "Pente montante (upsloping)",
        "flat": "Plate (flat)",
        "downsloping": "Pente descendante (downsloping)"
    }
    slope = st.selectbox("Pente du segment ST à l'effort", options=list(slope_options.keys()), format_func=lambda x: slope_options[x], index=1)

    submit_button = st.form_submit_button(label="🧪 Calculer le Diagnostic")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Prédiction & Visualisation des Résultats
# ─────────────────────────────────────────────────────────────────────────────

# Dataframe du patient
patient_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "dataset": dataset,
    "cp": cp,
    "trestbps": float(trestbps),
    "chol": float(chol),
    "fbs": str(fbs),
    "restecg": restecg,
    "thalch": float(thalch),
    "exang": str(exang),
    "oldpeak": float(oldpeak),
    "slope": slope
}])

# Calcul de prédiction
probabilities = pipeline.predict_proba(patient_df)[0]
prob_healthy = probabilities[0] * 100
prob_disease = probabilities[1] * 100
prediction = pipeline.predict(patient_df)[0]

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("🎯 Résultat du Diagnostic")
    
    if prediction == 1:
        st.markdown(f"""
            <div class="card-danger">
                <h2 style="margin:0; font-size:1.6rem; color:#ef4444;">⚠ RISQUE DE MALADIE CARDIAQUE DÉTECTÉ</h2>
                <p style="font-size:1.1rem; margin-top:10px; color:#fca5a5;">
                    Le modèle estime à <strong>{prob_disease:.1f}%</strong> la probabilité que ce patient présente une cardiopathie.
                </p>
                <hr style="border-color:#ef4444; opacity:0.3;">
                <p style="font-size:0.95rem; margin:0; color:#f87171;">
                    ⚕ <strong>Recommandation clinique :</strong> Un bilan cardiologique complémentaire (échocardiographie, épreuve d'effort approfondie ou coronarographie) est vivement recommandé.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="card-success">
                <h2 style="margin:0; font-size:1.6rem; color:#22c55e;">✓ PATIENT PROBABLEMENT SAIN</h2>
                <p style="font-size:1.1rem; margin-top:10px; color:#86efac;">
                    Le modèle estime à <strong>{prob_healthy:.1f}%</strong> la probabilité d'absence de maladie cardiaque.
                </p>
                <hr style="border-color:#22c55e; opacity:0.3;">
                <p style="font-size:0.95rem; margin:0; color:#4ade80;">
                    ⚕ <strong>Recommandation clinique :</strong> Aucune anomalie majeure n'est détectée. Poursuivre le suivi de routine et maintenir une bonne hygiène de vie.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Barre de progression personnalisée
    st.write(f"**Jauge de Risque Cardiaque : {prob_disease:.1f}%**")
    st.progress(int(prob_disease))

    # Tableau récapitulatif
    st.markdown("### 📊 Récapitulatif des Métriques Patient")
    summary_data = {
        "Paramètre": ["Âge", "Sexe", "Pression Artérielle", "Cholestérol", "Fréquence Cardiaque Max", "Dépression ST (oldpeak)", "Douleur Thoracique", "Angine Effort"],
        "Valeur": [f"{age} ans", sex, f"{trestbps} mm Hg", f"{chol} mg/dl", f"{thalch} bpm", f"{oldpeak}", cp_options[cp], exang_options[exang]],
        "Référence Normale": ["N/A", "N/A", "< 120 mm Hg", "< 200 mg/dl", f"~{220-age} bpm max", "< 1.0", "Asymptomatique", "Non"]
    }
    st.table(pd.DataFrame(summary_data))

with col_right:
    st.subheader("📈 Análise de Probabilité & Facteurs de Risque")
    
    # Diagramme circulaire de probabilité avec Matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    
    labels = ['Sain', 'Malade']
    sizes = [prob_healthy, prob_disease]
    colors = ['#22c55e', '#ef4444']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(  # type: ignore
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140,
        textprops=dict(color="white", weight="bold", size=11)
    )
    plt.setp(autotexts, size=12, weight="bold")
    ax.set_title("Répartition des Probabilités", color="#e2e8f0", fontsize=13, pad=12, fontweight="bold")
    
    st.pyplot(fig)

    # Explication des coefficients du modèle
    st.markdown("### 🔍 Facteurs Influents (Modèle Global)")
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocess"]
        cat_ohe = preprocessor.named_transformers_["cat"]
        
        num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
        cat_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope"]
        
        cat_features = cat_ohe.get_feature_names_out(cat_cols).tolist()
        all_features = num_cols + cat_features
        coefs = model.coef_[0]
        
        importance_df = pd.DataFrame({
            "Feature": all_features,
            "Coefficient": coefs,
            "Abs": np.abs(coefs)
        }).sort_values("Abs", ascending=False).head(6)
        
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor("#0f172a")
        ax2.set_facecolor("#0f172a")
        
        bar_colors = ['#ef4444' if c > 0 else '#22c55e' for c in importance_df["Coefficient"]]
        ax2.barh(importance_df["Feature"], importance_df["Coefficient"], color=bar_colors)
        ax2.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
        ax2.tick_params(colors="#e2e8f0")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#334155')
        ax2.spines['bottom'].set_color('#334155')
        ax2.set_title("Top 6 Facteurs Majeurs de Risque", color="#e2e8f0", fontsize=11, fontweight="bold")
        
        st.pyplot(fig2)
    except Exception as e:
        st.info("Visualisation des facteurs indisponible.")

# Export de rapport
st.markdown("---")
st.subheader("📥 Exportation du Diagnostic")
report_text = f"""==================================================
        RAPPORT DE DIAGNOSTIC CARDIOCHECK AI
==================================================
Date du bilan : {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
Patient : {age} ans, {sex}

RESULTATS DU DIAGNOSTIC :
- Classification : {"RISQUE DE MALADIE CARDIAQUE DÉTECTÉ" if prediction == 1 else "PATIENT PROBABLEMENT SAIN"}
- Probabilité de Maladie : {prob_disease:.2f}%
- Probabilité d'être Sain : {prob_healthy:.2f}%

VALEURS MESURÉES :
- Tension artérielle : {trestbps} mm Hg
- Cholestérol        : {chol} mg/dl
- Glycémie à jeun > 120 : {fbs}
- Douleur thoracique  : {cp}
- Angine d'effort     : {exang}
- Fréquence cardiaque max : {thalch} bpm
- Dépression ST       : {oldpeak}
- Pente ST            : {slope}

AVERTISSEMENT :
Ce document est généré par un système expert automatisé et doit être 
validé par un médecin spécialiste.
=================================================="""

st.download_button(
    label="📄 Télécharger le Rapport Médical (TXT)",
    data=report_text,
    file_name=f"rapport_cardio_patient_{age}ans_{sex}.txt",
    mime="text/plain"
)
