import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc

sns.set(style="whitegrid", font_scale=1.4)
plt.rcParams["font.family"] = "serif"

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame([
        {
            "label": int(item["label"]),
            **{k: int(v) for k, v in item["static_analysis"].items()}
        } for item in data
    ])

def plot_logistic_coefficients(result, feature_names):
    coefs = result.params[1:]  # exclude intercept
    pvals = result.pvalues[1:]

    fig, ax = plt.subplots(figsize=(8, 5))
    coefs.plot(kind='bar', ax=ax, color='royalblue', edgecolor='black')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Logistic Regression Coefficients", fontsize=16)
    ax.set_ylabel("Coefficient")
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    for i, p in enumerate(pvals):
        if p < 0.05:
            ax.text(i, coefs[i], "*", ha='center', va='bottom', fontsize=18, color='red')
    plt.tight_layout()
    plt.savefig("logistic_coefficients.pdf")
    plt.close()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of Combined Static Features")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.pdf")
    plt.close()

def plot_feature_distributions(df, feature_names):
    for feat in feature_names:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=feat, hue="label", palette="Set2", edgecolor='black')
        plt.title(f"Distribution of {feat} by Ponzi Label")
        plt.xlabel(f"{feat} (0 = Not Present, 1 = Present)")
        plt.ylabel("Count")
        plt.legend(title="Ponzi Label", labels=["Not Ponzi", "Ponzi"])
        plt.tight_layout()
        plt.savefig(f"distribution_{feat}.pdf")
        plt.close()

def main(analysis_resultes):
    df = load_data(analysis_resultes)
    feature_names = ["fund_flow", "profit_logic", "referral_mechanism", "withdrawal_control", "camouflage"]
    X = df[feature_names]
    X = sm.add_constant(X)
    y = df["label"]

    # Logistic Regression
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    # Visualization
    plot_logistic_coefficients(result, feature_names)
    plot_roc_curve(y, result.predict(X))
    plot_feature_distributions(df, feature_names)

    print("[INFO] All charts saved as .pdf")

if __name__ == "__main__":
    analysis_resultes = r''
    main(analysis_resultes)
