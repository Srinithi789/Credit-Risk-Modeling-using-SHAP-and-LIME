# Interpretable Machine Learning for Credit Risk Modeling using SHAP and LIME

This project builds an **interpretable machine learning pipeline** to predict **loan default risk** using an anonymized financial dataset (HELOC dataset).  
The primary focus is not only on model accuracy but also on **explainability**, using **SHAP** and **LIME** for transparent credit decision-making.

---

## 📌 Key Objectives

1. Build a robust ML model to classify applicants as **Good** or **Bad** credit risk.  
2. Apply **SHAP** for global and local interpretability:
   - Summary plots
   - Feature contributions
   - Waterfall & force plots  
3. Apply **LIME** to validate instance-level explanations.  
4. Compare SHAP vs LIME and interpret results for **loan officers** and **regulatory compliance**.  
5. Save all SHAP/LIME visualizations automatically.

---

## 📂 Project Structure

```
├── data/
│   └── heloc_dataset.csv
├── shap_plots/
│   ├── shap_summary_plot.png
│   ├── shap_waterfall_instance0.png
│   ├── shap_feature_importance.png
│   └── shap_dependence_*.png
├── lime_plots/
│   ├── lime_explanation_instance0.png
│   └── lime_instance_*.png
├── credit_risk_model.py
└── README.md
```

---

## 🧠 Dataset Description

The HELOC dataset contains:
- Credit bureau scores  
- Transaction summary data  
- Payment behaviours  
- Delinquency history  
- RiskPerformance label (Good = 1, Bad = 0)

Special missing values like **-7** and **-8** are treated as `NaN`.

---

## 🚀 Modeling Steps

### 1. Data Preprocessing
- Encode target (Good → 1, Bad → 0)
- Replace special missing values and fill using median
- Train/test split

### 2. Models Trained
The following models were evaluated:

| Model | Purpose |
|-------|----------|
| Random Forest | Baseline ensemble |
| Gradient Boosting | Stable boosting model |
| XGBoost | High performance boosting |
| LightGBM | Fast, scalable boosting |

Each model is evaluated by:
- Test accuracy  
- **5-fold Cross-validation accuracy**

The model with the highest CV accuracy becomes the **best model**.

---

## 📊 Feature Importance

The top 10 most important features from the best model are printed along with a SHAP bar plot.  
These features usually represent:
- Credit utilization  
- Delinquency months  
- Number of satisfactory trades  
- Inquiry counts  
- Revolving credit behavior

---

## 🔍 Explainability

### ✔ SHAP (SHapley Additive exPlanations)
Used for:
- **Global importance**  
- **Local explanation for each applicant**  
- **Regulatory-grade interpretability**

Plots generated:
- Summary plot  
- Feature importance bar plot  
- Waterfall plot  
- Force plot  
- Top-5 SHAP dependence plots  

### ✔ LIME (Local Interpretable Model-Agnostic Explanation)
Used to validate SHAP’s local explanations.  
Plots are saved for the first 10 test instances.

---

## 🆚 SHAP vs LIME Summary

| Aspect | SHAP | LIME |
|--------|-------|-------|
| Nature | Game-theory exact | Local approximation |
| Consistency | Guaranteed | Not guaranteed |
| Best for | Regulatory, fairness audits | Debugging individual cases |
| Speed | Slower | Faster |
| Explanation type | Additive contributions | Local linear model |

---

## 📁 Automatic Saving of Plots

All plots are saved automatically into:

```
shap_plots/
lime_plots/
```

This includes:
- SHAP summary
- SHAP waterfall
- SHAP force
- SHAP dependence
- LIME instance-wise plots

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
python credit_risk_model.py
```

---

## ✅ Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
lime
shap
matplotlib
```

Install them via:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm lime shap matplotlib
```

---

## 📘 Interpretation for Loan Officers

✔ Identify why an applicant was classified as **Good** or **Bad**  
✔ See which financial behaviours contributed positively or negatively  
✔ Use explanations to support **fair & transparent** lending decisions  
✔ Comply with **AI fairness and regulatory norms**

---

## 📌 Final Outcome

This project provides:
- A complete ML pipeline  
- Deep interpretability using SHAP & LIME  
- Automated visualization export  
- A framework suitable for:
  - Research  
  - Banking applications  
  - Audit reporting  
  - Academic submissions  

---

## 📞 Contact
Maintained by **Srinithinarayanan**  
For guidance or improvements, feel free to reach out!

---
