import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ***********************************************************
# Step 2.4: Churn vs. Non-churn Comparison
# ***********************************************************

# 2.4 Comparing churn vs non-churn across numerical features
def plot_churn_numerical_comparison(train_eda_outliers_removed, num_cols):
    # Boxplots by churn status
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    for variable, subplot in zip(num_cols, ax.flatten()):
        sns.boxplot(x="churn_risk_score", y=variable, data=train_eda_outliers_removed, ax=subplot)
        subplot.set_title(f"{variable} by Churn Status")
        subplot.set_xlabel("churn_risk_score")
        subplot.set_ylabel(variable)
    for i in range(len(num_cols), ax.size):
        fig.delaxes(ax.flatten()[i])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_churn_boxplots_numerical.png"), bbox_inches='tight')
    plt.close()

    # KDE plots by churn status
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    for variable, subplot in zip(num_cols, ax.flatten()):
        sns.kdeplot(data=train_eda_outliers_removed, x=variable,
                    hue='churn_risk_score', fill=True, ax=subplot)
        subplot.set_title(f"{variable} by Churn Status")
        subplot.set_xlabel(variable)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_churn_kde_numerical.png"), bbox_inches='tight')
    plt.close()

    print("\n[plot_churn_numerical_comparison] Churn numerical comparison plots saved.")


# 2.4 Comparing churn vs non-churn across categorical features
def plot_churn_categorical_comparison(train_eda_outliers_removed, cat_cols):
    # Countplots
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(18, 18))
    axes = ax.flatten()
    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, hue="churn_risk_score", data=train_eda_outliers_removed, ax=axes[i])
        axes[i].set_title(f"{col} by Churn")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].legend(title="Churn", loc="upper right")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_churn_countplots_categorical.png"), bbox_inches='tight')
    plt.close()

    # Heatmaps
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(18, 18))
    axes = ax.flatten()
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(
            train_eda_outliers_removed[col],
            train_eda_outliers_removed["churn_risk_score"],
            normalize="index"
        ) * 100
        sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"{col} vs Churn (%)")
    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_churn_heatmaps_categorical.png"), bbox_inches='tight')
    plt.close()

    print("\n[plot_churn_categorical_comparison] Churn categorical comparison plots saved.")


# 2.4 Violin plots for churn-sensitive feature combinations
def plot_violin_churn_comparisons(train_eda_outliers_removed):
    def plot_violin_churn(df, feature, filename):
        churn_sensitive_num_features = ['avg_time_spent', 'avg_transaction_value',
                                        'avg_frequency_login_days', 'points_in_wallet']
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        for var, subplot in zip(churn_sensitive_num_features, ax.flatten()):
            sns.violinplot(data=df, x=feature, y=var, hue="churn_risk_score",
                           split=True, ax=subplot)
            subplot.set_title(f"{var} by {feature} and Churn")
            subplot.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
        plt.close()

    # Membership tiers
    plot_violin_churn(train_eda_outliers_removed, 'membership_category',
                      "eda_violin_membership.png")

    # Negative feedback
    plot_violin_churn(train_eda_outliers_removed, 'feedback',
                      "eda_violin_feedback.png")

    print("\n[plot_violin_churn_comparisons] Violin plots saved.")


# ***********************************************************
# Step 2.5: Correlation Matrix
# ***********************************************************

# 2.5 Label encoding EDA datasets for correlation analysis
def apply_label_mappings(train_eda_clean, train_eda_outliers_removed, train_eda_imputed, cat_cols):
    # Create mapping dictionary
    mappings = {
        "gender": {'F': 0, 'M': 1},
        "region_category": {'City': 0, 'Town': 1, 'Village': 2},
        "membership_category": {'No Membership': 0, 'Basic Membership': 1,
                                'Silver Membership': 2, 'Gold Membership': 3,
                                'Premium Membership': 4, 'Platinum Membership': 5},
        "joined_through_referral": {'Yes': 1, 'No': 0},
        "preferred_offer_types": {'Without Offers': 0, 'Gift Vouchers/Coupons': 1,
                                  'Credit/Debit Card Offers': 2},
        "medium_of_operation": {'Desktop': 0, 'Smartphone': 1, 'Both': 2},
        "internet_option": {'Mobile_Data': 0, 'Wi-Fi': 1, 'Fiber_Optic': 2},
        "used_special_discount": {'Yes': 1, 'No': 0},
        "offer_application_preference": {'Yes': 1, 'No': 0},
        "past_complaint": {'Yes': 1, 'No': 0},
        "complaint_status": {'Not Applicable': 0, 'No Information Available': 1,
                             'Solved': 2, 'Solved in Follow-up': 3, 'Unsolved': 4},
        "feedback": {'Too many ads': -1, 'Poor Website': -1,
                     'Poor Customer Service': -1, 'Poor Product Quality': -1,
                     'No reason specified': 0, 'Products always in Stock': 1,
                     'Quality Customer Care': 1, 'User Friendly Website': 1,
                     'Reasonable Price': 1}
    }

    def apply_mappings(df, maps):
        for col, mapping in maps.items():
            if col in df.columns:
                df[f"{col}_label"] = df[col].map(mapping)
        return df

    eda_sets = [train_eda_clean, train_eda_outliers_removed, train_eda_imputed]
    for i, df in enumerate(eda_sets):
        eda_sets[i] = apply_mappings(df, mappings)

    # Check if mappings are correct
    for col in cat_cols:
        if col in train_eda_clean.columns and f"{col}_label" in train_eda_clean.columns:
            df_check = pd.crosstab(train_eda_clean[col], train_eda_clean[f"{col}_label"], dropna=False)
            print(df_check, '\n')

    print("\n[apply_label_mappings] Label mappings applied for correlation analysis.")
    return eda_sets[0], eda_sets[1], eda_sets[2]


# 2.5 Plotting correlation matrices
def plot_correlation_matrices(train_eda_clean, train_eda_outliers_removed, train_eda_imputed, num_cols):
    label_cols = [col for col in train_eda_clean.columns if col.endswith("_label")]
    selected_cols = label_cols + list(num_cols) + ["churn_risk_score"]

    def get_correlation_matrix(df, cols):
        return df[cols].corr()

    def plot_corr_heatmap(corr, figsize=(18, 14), title="Correlation Matrix", filename="corr.png"):
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=figsize)
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                    linewidths=0.4, annot=True, annot_kws={"size": 12},
                    fmt=".2f", cbar_kws={"shrink": 0.8})
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
        plt.close()

    # Before removing outliers
    correlation_matrix_clean = get_correlation_matrix(train_eda_clean, selected_cols)
    plot_corr_heatmap(correlation_matrix_clean, title="Correlation Matrix Raw (Lower Triangle)",
                      filename="eda_correlation_raw.png")

    # After removing outliers and missing values
    train_eda_no_NA = train_eda_outliers_removed.dropna()
    print("Shape after dropping NaN for correlation:", train_eda_no_NA.shape)
    correlation_matrix_no_NA = get_correlation_matrix(train_eda_no_NA, selected_cols)
    plot_corr_heatmap(correlation_matrix_no_NA,
                      title="Correlation Matrix After Outlier and Missing Value Removal (Lower Triangle)",
                      filename="eda_correlation_outlier_removed.png")

    # After outlier removal and imputation
    correlation_matrix_imputed = get_correlation_matrix(train_eda_imputed, selected_cols)
    plot_corr_heatmap(correlation_matrix_imputed,
                      title="Correlation Matrix After Outlier Removal and Imputation (Lower Triangle)",
                      filename="eda_correlation_imputed.png")

    # Correlation with target only
    def plot_corr_with_target(correlation_matrix, ax, title):
        corr_with_target = correlation_matrix["churn_risk_score"].sort_values(ascending=False)
        sns.heatmap(corr_with_target.to_frame(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title(title)

    fig, axes = plt.subplots(1, 3, figsize=(18, 12), dpi=150)
    plot_corr_with_target(correlation_matrix_clean, axes[0], "Correlation with Churn Raw")
    plot_corr_with_target(correlation_matrix_no_NA, axes[1], "After Outlier and Missing Value Removal")
    plot_corr_with_target(correlation_matrix_imputed, axes[2], "After Outlier Removal and Imputation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_with_target.png"), bbox_inches='tight')
    plt.close()

    print("\n[plot_correlation_matrices] All correlation matrix charts saved.")
    return correlation_matrix_clean, correlation_matrix_no_NA, correlation_matrix_imputed


# ***********************************************************
# Pipeline orchestrators
# ***********************************************************

def run_churn_comparison(X_train_eda_clean, X_train_eda_outliers_removed,
                         X_train_eda_imputed, y_train_eda, num_cols, binary_cols, nominal_cols):
    print("\n" + "*" * 60)
    print("STEP 2.4: CHURN VS NON-CHURN COMPARISON")
    print("*" * 60)

    y_train_eda_named = y_train_eda.rename("churn_risk_score")

    train_eda_clean = pd.concat([X_train_eda_clean, y_train_eda_named], axis=1)
    train_eda_outliers_removed = pd.concat([X_train_eda_outliers_removed, y_train_eda_named], axis=1)
    train_eda_imputed = pd.concat([X_train_eda_imputed, y_train_eda_named], axis=1)

    cat_cols = binary_cols + nominal_cols

    plot_churn_numerical_comparison(train_eda_outliers_removed, num_cols)
    plot_churn_categorical_comparison(train_eda_outliers_removed, cat_cols)
    plot_violin_churn_comparisons(train_eda_outliers_removed)

    print("\n[run_churn_comparison] Step 2.4 completed.")
    return train_eda_clean, train_eda_outliers_removed, train_eda_imputed


def run_correlation_analysis(X_train_eda_clean, X_train_eda_outliers_removed,
                              X_train_eda_imputed, y_train_eda, num_cols, binary_cols, nominal_cols):
    print("\n" + "*" * 60)
    print("STEP 2.5: CORRELATION MATRIX")
    print("*" * 60)

    y_train_eda_named = y_train_eda.rename("churn_risk_score")

    train_eda_clean = pd.concat([X_train_eda_clean, y_train_eda_named], axis=1)
    train_eda_outliers_removed = pd.concat([X_train_eda_outliers_removed, y_train_eda_named], axis=1)
    train_eda_imputed = pd.concat([X_train_eda_imputed, y_train_eda_named], axis=1)

    cat_cols = binary_cols + nominal_cols

    train_eda_clean, train_eda_outliers_removed, train_eda_imputed = apply_label_mappings(
        train_eda_clean, train_eda_outliers_removed, train_eda_imputed, cat_cols
    )

    plot_correlation_matrices(train_eda_clean, train_eda_outliers_removed, train_eda_imputed, num_cols)

    print("\n[run_correlation_analysis] Step 2.5 completed.")
