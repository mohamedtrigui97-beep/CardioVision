import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif : sauvegarde sans ouvrir de fenetre
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ─────────────────────────────────────────────
#  1. Chargement
# ─────────────────────────────────────────────
df = pd.read_csv("data/heart_disease_clean.csv")
print("=" * 60)
print("  REGRESSION LOGISTIQUE — Heart Disease UCI")
print("=" * 60)
print(f"  Dataset : {df.shape[0]} lignes x {df.shape[1]} colonnes")

# ─────────────────────────────────────────────
#  2. Préparation des features et cible
# ─────────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]

# Identifier les colonnes catégorielles et numériques
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()

print(f"\n  Features numériques  : {num_cols}")
print(f"  Features catégorielles: {cat_cols}")
print(f"  Cible (target) : 0=Sain ({(y==0).sum()}), 1=Malade ({(y==1).sum()})")

# ─────────────────────────────────────────────
#  3. Split Train / Test
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train : {len(X_train)} exemples")
print(f"  Test  : {len(X_test)} exemples")

# ─────────────────────────────────────────────
#  4. Pipeline de prétraitement + modèle
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
])

# ─────────────────────────────────────────────
#  5. Entraînement
# ─────────────────────────────────────────────
pipeline.fit(X_train, y_train)
print("\n  Modèle entraîné avec succès !")

# ─────────────────────────────────────────────
#  6. Évaluation
# ─────────────────────────────────────────────
y_pred      = pipeline.predict(X_test)
y_prob      = pipeline.predict_proba(X_test)[:, 1]
accuracy    = accuracy_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_prob)
cv_scores   = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print("\n" + "=" * 60)
print("  RÉSULTATS")
print("=" * 60)
print(f"  Accuracy  (test)    : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  AUC-ROC   (test)    : {auc:.4f}")
print(f"  CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\n  Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Sain (0)", "Malade (1)"]))

# ─────────────────────────────────────────────
#  7. Noms des features après encoding
# ─────────────────────────────────────────────
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

coefs = pipeline.named_steps["model"].coef_[0]
feature_importance = pd.DataFrame({
    "Feature": all_feature_names,
    "Coefficient": coefs,
    "Abs": np.abs(coefs)
}).sort_values("Abs", ascending=False)

print("\n  Top 10 features les plus importantes :")
print(feature_importance.head(10).to_string(index=False))

# ─────────────────────────────────────────────
#  8. Visualisations
# ─────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0a0f1e")

# ── Couleurs
C_GREEN  = "#00e676"
C_RED    = "#ff3d71"
C_BLUE   = "#3d9bff"
C_GOLD   = "#ffd740"
C_BG     = "#0a0f1e"
C_PANEL  = "#111827"
C_TEXT   = "#dce6ff"
C_GRID   = "#1a2540"

def panel(ax):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.grid(color=C_GRID, linestyle="--", linewidth=0.5)

# ── 8a. Matrice de confusion
ax1 = fig.add_subplot(2, 3, 1)
panel(ax1)
cm = confusion_matrix(y_test, y_pred)
im = ax1.imshow(cm, cmap="Blues", interpolation="nearest")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                 fontsize=22, fontweight="bold",
                 color="white" if cm[i, j] > cm.max()/2 else C_TEXT)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(["Prédit Sain", "Prédit Malade"], fontsize=10, color=C_TEXT)
ax1.set_yticklabels(["Réel Sain", "Réel Malade"], fontsize=10, color=C_TEXT)
ax1.set_title("Matrice de Confusion", fontsize=13, fontweight="bold", pad=12)

# ── 8b. Courbe ROC
ax2 = fig.add_subplot(2, 3, 2)
panel(ax2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, color=C_GREEN, linewidth=2.5, label=f"AUC = {auc:.3f}")
ax2.plot([0, 1], [0, 1], color=C_GRID, linewidth=1.5, linestyle="--", label="Aléatoire")
ax2.fill_between(fpr, tpr, alpha=0.1, color=C_GREEN)
ax2.set_xlabel("Taux Faux Positifs", fontsize=11)
ax2.set_ylabel("Taux Vrais Positifs", fontsize=11)
ax2.set_title("Courbe ROC", fontsize=13, fontweight="bold", pad=12)
ax2.legend(facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=11)

# ── 8c. Coefficients (Top 12)
ax3 = fig.add_subplot(2, 3, 3)
panel(ax3)
top12 = feature_importance.head(12).sort_values("Coefficient")
colors = [C_GREEN if v > 0 else C_RED for v in top12["Coefficient"]]
bars = ax3.barh(top12["Feature"], top12["Coefficient"], color=colors, height=0.6)
ax3.axvline(0, color=C_TEXT, linewidth=0.8, linestyle="--")
ax3.set_title("Coefficients (Top 12)", fontsize=13, fontweight="bold", pad=12)
ax3.set_xlabel("Coefficient", fontsize=11)
pos_patch = mpatches.Patch(color=C_GREEN, label="Risque accru (+)")
neg_patch = mpatches.Patch(color=C_RED,   label="Risque réduit (−)")
ax3.legend(handles=[pos_patch, neg_patch], facecolor=C_PANEL,
           edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=9)

# ── 8d. Distribution des probabilités prédites
ax4 = fig.add_subplot(2, 3, 4)
panel(ax4)
prob_sain   = y_prob[y_test == 0]
prob_malade = y_prob[y_test == 1]
ax4.hist(prob_sain,   bins=20, color=C_BLUE,  alpha=0.7, label="Réel Sain",   edgecolor="none")
ax4.hist(prob_malade, bins=20, color=C_RED,   alpha=0.7, label="Réel Malade", edgecolor="none")
ax4.axvline(0.5, color=C_GOLD, linewidth=1.5, linestyle="--", label="Seuil 0.5")
ax4.set_xlabel("Probabilité prédite (Malade)", fontsize=11)
ax4.set_ylabel("Fréquence", fontsize=11)
ax4.set_title("Distribution des Probabilités", fontsize=13, fontweight="bold", pad=12)
ax4.legend(facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=10)

# ── 8e. Cross-validation scores
ax5 = fig.add_subplot(2, 3, 5)
panel(ax5)
folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
bar_colors = [C_GREEN if s >= cv_scores.mean() else C_BLUE for s in cv_scores]
bars = ax5.bar(folds, cv_scores, color=bar_colors, width=0.5, edgecolor="none")
ax5.axhline(cv_scores.mean(), color=C_GOLD, linewidth=2, linestyle="--",
            label=f"Moyenne = {cv_scores.mean():.3f}")
for bar, val in zip(bars, cv_scores):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10, color=C_TEXT)
ax5.set_ylim(0.7, 1.0)
ax5.set_title("Cross-Validation (5 Folds)", fontsize=13, fontweight="bold", pad=12)
ax5.set_ylabel("Accuracy", fontsize=11)
ax5.legend(facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=10)

# ── 8f. Métriques récapitulatives
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor(C_PANEL)
ax6.axis("off")
for spine in ax6.spines.values():
    spine.set_edgecolor(C_GRID)

from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

metrics = [
    ("Accuracy",  f"{accuracy*100:.1f}%", C_GREEN),
    ("AUC-ROC",   f"{auc:.4f}",           C_BLUE),
    ("Précision", f"{precision*100:.1f}%", C_GOLD),
    ("Rappel",    f"{recall*100:.1f}%",    C_RED),
    ("F1-Score",  f"{f1:.4f}",            "#b388ff"),
    ("CV Mean",   f"{cv_scores.mean()*100:.1f}%", "#80cbc4"),
]
ax6.text(0.5, 0.95, "Récapitulatif des Métriques", ha="center", va="top",
         fontsize=13, fontweight="bold", color=C_TEXT, transform=ax6.transAxes)
for i, (name, val, color) in enumerate(metrics):
    y_pos = 0.80 - i * 0.13
    ax6.add_patch(plt.Rectangle((0.05, y_pos - 0.04), 0.9, 0.11,
                                 facecolor=color+"22", edgecolor=color+"55",
                                 transform=ax6.transAxes, clip_on=False, linewidth=1))
    ax6.text(0.15, y_pos + 0.015, name, ha="left", va="center",
             fontsize=11, color=C_TEXT, transform=ax6.transAxes)
    ax6.text(0.85, y_pos + 0.015, val, ha="right", va="center",
             fontsize=13, fontweight="bold", color=color, transform=ax6.transAxes)

# ── Titre global
fig.suptitle("Régression Logistique — Heart Disease UCI",
             fontsize=18, fontweight="bold", color=C_TEXT, y=1.01)

plt.tight_layout()
plt.savefig("data/logistic_regression_results.png", dpi=150,
            bbox_inches="tight", facecolor=C_BG)
plt.close()
print("\n  Graphique sauvegarde -> data/logistic_regression_results.png")
print("=" * 60)
