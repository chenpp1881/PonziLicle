import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def load_jsonl_data(file_path):
    records = []

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for entry in raw_data:
            try:
                if "explanations" in entry and "qualitative" in entry["explanations"]:
                    q = entry["explanations"]["qualitative"]
                    records.append({
                        "index": entry.get("index", None),
                        "label": entry.get("label", None),
                        "fund_flow": 1 if q["fund_flow"].lower() == "yes" else 0,
                        "profit_logic": 1 if q["profit_logic"].lower() == "yes" else 0,
                        "referral_mechanism": 1 if q["referral_mechanism"].lower() == "yes" else 0,
                        "withdrawal_control": 1 if q["withdrawal_control"].lower() == "yes" else 0,
                        "camouflage": 1 if q["camouflage"].lower() == "yes" else 0,
                    })
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def full_statistical_analysis(df):
    features = ["fund_flow", "profit_logic", "referral_mechanism", "withdrawal_control", "camouflage"]
    X = df[features]
    y = df["label"]

    # === 2. 卡方检验 ===
    chi2_results = []
    for feature in features:
        contingency_table = pd.crosstab(X, y)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_results.append({
            "Feature": feature,
            "Chi2 Statistic": chi2,
            "p-value": p,
            "Significant (p < 0.05)": "✓" if p < 0.05 else "✗"
        })

    # 构建 DataFrame，并把 Feature 设为行索引
    chi_df = pd.DataFrame(chi2_results).set_index("Feature")

    # 只画 p-value，一列的热力图，纵轴就是 Feature
    plt.figure(figsize=(6, len(features) * 0.4))  # 根据特征数量动态调整高度
    sns.heatmap(
        chi_df[["p-value"]],
        annot=True,
        cmap="Blues_r",
        cbar=True,
        yticklabels=True
    )
    plt.title("p-value Heatmap from Chi-Square Test")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # === 3. 卡方热力图展示 ===
    plt.figure(figsize=(6, 4))
    sns.heatmap(chi_df[["Chi2 Statistic"]], annot=True, cmap="YlOrRd", cbar=True)
    plt.title("Chi-Square Statistics per Feature")
    plt.tight_layout()
    plt.show()

def formalize_column_names(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    """
    修改DataFrame的列名：如果旧列名存在，则改为新列名。

    参数：
        df (pd.DataFrame): 原始DataFrame
        rename_dict (dict): 映射字典，key为旧列名，value为新列名（形式化后的）

    返回：
        pd.DataFrame: 修改列名后的DataFrame
    """
    rename_mapping = {old: new for old, new in rename_dict.items() if old in df.columns}
    return df.rename(columns=rename_mapping)


def compute_chi2_cramers(df):
    """
    计算每个特征与标签之间的卡方统计量、自由度、p-value 和 Cramér's V。

    参数:
        df (pd.DataFrame): 包含特征和标签的数据框
        features (list of str): 特征列名列表
        label (str): 标签列名，默认为 'label'

    返回:
        pd.DataFrame: 包含 ['Feature','Chi2 Statistic','df','p-value',"Cramér's V"] 列的结果表，按 Chi2 值降序排序
    """
    features = ["fund_flow", "profit_logic", "referral_mechanism", "withdrawal_control", "camouflage"]
    label = df["label"]
    results = []
    old_to_formal_name = {
        "fund_flow": "Fund Flow",
        "profit_logic": "Profit Logic",
        "referral_mechanism": "Referral Mechanism",
        "withdrawal_control": "Withdrawal Control",
        "camouflage": "Camouflage Naming"
    }
    df = formalize_column_names(df, old_to_formal_name)
    features = list(old_to_formal_name.values())
    for feature in features:
        contingency = pd.crosstab(df[feature], label)
        chi2, p, dof, _ = chi2_contingency(contingency)
        n = contingency.values.sum()
        r, k = contingency.shape
        cramers_v = np.sqrt(chi2 / (n * min(r-1, k-1)))

        results.append({
            'Feature': feature,
            'Chi2 Statistic': chi2,
            'df': dof,
            'p-value': p,
            "Cramér's V": cramers_v
        })
    results_df = pd.DataFrame(results).sort_values('Chi2 Statistic', ascending=False)
    results_df.to_excel('chi2_results.xlsx', index=False)

    # 1. 预处理：计算 -log10(p-value) 并标记显著性
    results_df['neg_log10_p'] = -np.log10(results_df['p-value'])
    results_df['significant'] = results_df['p-value'] < 0.05

    # 2. 排序：按显著性及 neg_log10_p 排序，重要特征靠前
    plot_df = (results_df
               .sort_values(['significant', 'neg_log10_p'],
                            ascending=[False, False]))

    # 3. 全局风格设置
    sns.set_style("white")  # 纯白背景
    sns.set_context("paper", font_scale=1.3)  # 论文级字号
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 0.8  # 轴线加粗
    plt.rcParams["xtick.direction"] = "out"  # 刻度朝外
    plt.rcParams["ytick.direction"] = "out"

    # 4. 作图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 单色条形，显著性用深浅区分
    palette = ['#7f7f7f' if not sig else '#1f77b4'
               for sig in plot_df['significant']]

    bars = ax.bar(
        x=plot_df['Feature'],
        height=plot_df['neg_log10_p'],
        color=palette,
        edgecolor='black',
        linewidth=0.05
    )

    # 5. 标注每个条的数值
    for bar, val in zip(bars, plot_df['neg_log10_p']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x 位置居中
            val + 0.05,  # y 值略上方
            f"{val:.2f}",
            ha='center',
            va='bottom',
            fontsize=12
        )

    # 6. 显著性阈值线
    threshold = -np.log10(0.05)
    ax.axhline(threshold, color='black', linestyle='--', linewidth=1)
    ax.text(
        len(plot_df) - 0.05, threshold + 0.05,
        "p = 0.05",
        ha='right',
        va='bottom',
        fontsize=12,
        color='black'
    )

    # 7. 坐标轴与标题
    ax.set_xlabel("")  # x 轴标签为空，由下方注释说明
    ax.set_ylabel("-log10(p-value)", fontsize=14)
    # ax.set_ylabel(r"$-\log_{10}(p\text{-value})$")
    # ax.set_title("Feature Significance by Chi–Square Test", fontsize=14, pad=15)

    # 8. 细节优化
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['Feature'], rotation=45, ha='right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)
    ax.tick_params(axis='x', labelsize=12)

    # 9. 图例说明（手动添加）
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='p < 0.05'),
        mpatches.Patch(facecolor='#7f7f7f', edgecolor='black', label='p ≥ 0.05')
    ]
    ax.legend(
        handles=legend_patches,
        title="Significance",
        loc='upper right',
        frameon=False,
        fontsize=13,
        title_fontsize=11
    )

    plt.tight_layout()
    plt.savefig("chi2_pvalue_barplot.png", dpi=300, bbox_inches="tight")
    plt.show()

# ------------------------
# 使用示例（请确保 df 和 features 已定义）:
# ------------------------
# features = ['featureA', 'featureB', 'featureC', 'featureD', 'featureE']
# results_df = compute_chi2_cramers(df, features, label='label')
# import ace_tools as tools; tools.display_dataframe_to_user("Chi-Square Test Results", results_df)




# Re-import required modules after environment reset
# import json
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier, plot_importance
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
#
# def model_with_xgboost_metrics(df):
#     features = ["fund_flow", "profit_logic", "referral_mechanism", "withdrawal_control", "camouflage"]
#     X = df[features].to_numpy()
#     y = df["label"].to_numpy()
#
#     # 数据划分
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # 模型训练
#     model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
#     model.fit(X_train, y_train)
#
#     # 测试集预测 + 分类指标
#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     report_df = pd.DataFrame(report).T[["precision", "recall", "f1-score"]]
#
#     # 打印指标
#     print("=== Classification Report on Test Set ===")
#     print(report_df)
#
#     # 可视化特征重要性
#     plt.figure(figsize=(8, 4))
#     plot_importance(model, importance_type="weight", xlabel="Feature Importance", title="XGBoost Feature Importance")
#     plt.tight_layout()
#     plt.show()
#
#     return report_df


import json

def calculate_camouflage_ratio(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_ponzi = 0
    ponzi_with_camouflage = 0

    for item in data:
        label = item.get("label", 0)
        camouflage = item.get("explanations", {}).get("qualitative", "").get("camouflage", "").strip().lower()

        if label == 1 or label == "ponzi":  # 支持数字或字符串标注
            total_ponzi += 1
            if camouflage == "yes":
                ponzi_with_camouflage += 1

    if total_ponzi == 0:
        print("No Ponzi contracts found.")
        return

    ratio = ponzi_with_camouflage / total_ponzi
    print(f"💡 Total Ponzi contracts        : {total_ponzi}")
    print(f"🔍 With naming camouflage       : {ponzi_with_camouflage}")
    print(f"📊 Camouflage ratio (Ponzi only): {ratio:.4f} ({ratio * 100:.2f}%)")





if __name__ == '__main__':
    data_path = r'llm_explanations.json'
    datasets = load_jsonl_data(data_path)
    # full_statistical_analysis(datasets)
    compute_chi2_cramers(datasets)
    # model_with_xgboost_metrics(datasets)

    # calculate_camouflage_ratio(data_path)