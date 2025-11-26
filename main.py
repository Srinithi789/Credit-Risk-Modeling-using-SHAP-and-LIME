# ============================================================
#     Interpretable Machine Learning for Credit Risk
#     SHAP + LIME on HELOC Dataset
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
#                   1. LOAD & CLEAN DATA
# ------------------------------------------------------------
df = pd.read_csv("archive_2/heloc_dataset_v1 (1).csv")

# Encode target
df["RiskPerformance"] = df["RiskPerformance"].map({"Good": 1, "Bad": 0})

# Replace special missing values
df.replace([-7, -8], np.nan, inplace=True)

# Fill missing with median
df.fillna(df.median(), inplace=True)

# Split
from sklearn.model_selection import train_test_split
X = df.drop("RiskPerformance", axis=1)
y = df["RiskPerformance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=22, test_size=0.2
)

# ------------------------------------------------------------
#         2. TRAIN MULTIPLE MODELS + CROSS VALIDATION
# ------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    "LightGBM": LGBMClassifier(verbose=-1)
}

print("\n============================")
print(" MODEL ACCURACY (TEST SET)  ")
print("============================")

test_accuracies = {}
cv_accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    test_accuracies[name] = acc
    print(f"{name}: {acc:.4f}")

print("\n============================")
print("     5-FOLD CROSS VALIDATION ")
print("============================")

for name, model in models.items():
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    cv_accuracies[name] = cv.mean()
    print(f"{name} CV Accuracy: {cv.mean():.4f}")

# Best model based on CV
best_model_name = max(cv_accuracies, key=cv_accuracies.get)
best_model = models[best_model_name]

print("\nBest Model Based on CV Accuracy:", best_model_name)

# ------------------------------------------------------------
#                    3. FEATURE IMPORTANCE
# ------------------------------------------------------------
print("\n============================")
print("       FEATURE IMPORTANCE    ")
print("============================")

best_model.fit(X_train, y_train)

importances = best_model.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feat_imp.head(10))

# ------------------------------------------------------------
#                       4. SHAP ANALYSIS
# ------------------------------------------------------------
import shap
shap_explainer = shap.Explainer(best_model, X_train)
shap_values = shap_explainer(X_test)
print("shap_values.values shape:", np.array(shap_values.values).shape)
print("shap_values.base_values shape:", np.array(shap_values.base_values).shape)

# --- SHAP Waterfall ---
shap_row = shap_values[0]       # this is a SHAP object
shap.plots.waterfall(shap_row)


# ------------------------------------------------------------
#            AUTO-SUMMARY TEXT: SHAP INTERPRETATION
# ------------------------------------------------------------
def interpret_shap(instance_idx):
    print("\n============================")
    print("       SHAP INTERPRETATION  ")
    print("============================")

    row_vals = shap_values[instance_idx].values
    feature_names = X_test.columns

    pos = []
    neg = []

    for f, v in zip(feature_names, row_vals):
        if v > 0:
            pos.append((f, v))
        else:
            neg.append((f, v))

    pos = sorted(pos, key=lambda x: x[1], reverse=True)
    neg = sorted(neg, key=lambda x: x[1])

    print("\nFeatures pushing prediction TOWARDS Good (1):")
    for f, v in pos[:5]:
        print(f"{f}: {v:.4f}")

    print("\nFeatures pushing prediction TOWARDS Bad (0):")
    for f, v in neg[:5]:
        print(f"{f}: {v:.4f}")

interpret_shap(0)


# ------------------------------------------------------------
#                        5. LIME ANALYSIS
# ------------------------------------------------------------
def predict_fn_with_names(x):
    df_input = pd.DataFrame(x, columns=X_train.columns)
    return best_model.predict_proba(df_input)

from lime.lime_tabular import LimeTabularExplainer

lime_exp = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Bad", "Good"],
    discretize_continuous=True
)

lime_explanation = lime_exp.explain_instance(
    data_row=X_test.values[0],
    predict_fn=predict_fn_with_names
)


# --- LIME Output Text ---
print("\n============================")
print("       LIME INTERPRETATION  ")
print("============================")

for item in lime_explanation.as_list():
    print(item)

# ------------------------------------------------------------
#                   6. SHAP vs LIME SUMMARY
# ------------------------------------------------------------
print("\n============================")
print("   SHAP vs LIME COMPARISON  ")
print("============================")

print("""
SHAP:
  - Provides mathematically exact Shapley values.
  - Best for global feature importance.
  - Highly suitable for regulatory explanations.

LIME:
  - Local approximation around an instance.
  - Good for debugging single predictions.
  - Helps verify SHAP explanations in borderline cases.
""")

# ------------------------------------------------------------
#                       7. SAVE ALL PLOTS
# ------------------------------------------------------------
import os
import matplotlib.pyplot as plt

os.makedirs("shap_plots", exist_ok=True)
os.makedirs("lime_plots", exist_ok=True)

print("\nSaving SHAP plots...")

# SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_plots/shap_summary_plot.png", dpi=300)
plt.close()

# SHAP Bar Plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig("shap_plots/shap_feature_importance.png", dpi=300)
plt.close()

# SHAP Waterfall
plt.figure()
shap_row = shap_values[0]       # this is a SHAP object
shap.plots.waterfall(shap_row)
plt.savefig("shap_plots/shap_waterfall_instance0.png", dpi=300)
plt.close()

print("SHAP plots saved.")

# ----------------------- LIME Save ---------------------------
print("\nSaving LIME plots...")

lime_fig = lime_explanation.as_pyplot_figure()
lime_fig.savefig("lime_plots/lime_explanation_instance0.png", dpi=300)
plt.close(lime_fig)

# ------------------------------------------------------------
#    EXTRA SHAP & LIME (Force, Dependence, Multiple Samples)
# ------------------------------------------------------------
print("\nSaving additional SHAP & LIME plots...")

# SHAP Force Plot
shap.force_plot(
    shap_explainer.expected_value,
    shap_values[0].values,
    X_test.iloc[0],
    matplotlib=True,
    show=False
)
plt.savefig("shap_plots/shap_force_plot_instance0.png", dpi=300, bbox_inches='tight')
plt.close()

# SHAP Dependence Plots
top5 = feat_imp["Feature"].head(5).tolist()

for f in top5:
    plt.figure()
    shap.dependence_plot(f, shap_values.values, X_test, show=False)
    plt.savefig(f"shap_plots/shap_dependence_{f}.png", dpi=300)
    plt.close()

# LIME plots for first 10 samples
for i in range(10):
    lime_exp_i = lime_exp.explain_instance(
        data_row=X_test.values[i],
        predict_fn=predict_fn_with_names
    )
    fig_i = lime_exp_i.as_pyplot_figure()
    fig_i.savefig(f"lime_plots/lime_instance_{i}.png", dpi=300)
    plt.close(fig_i)

print("All plots saved successfully!")
