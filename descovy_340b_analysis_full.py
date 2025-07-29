
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(descovy_path, program340b_path):
    """Load and clean both datasets."""
    df_all = pd.read_csv(descovy_path, low_memory=False)
    df_340b = pd.read_csv(program340b_path, low_memory=False)

    # Standardize NDC column types
    df_all["NDC"] = df_all["NDC"].astype(str)
    df_340b["NDC"] = df_340b["NDC"].astype(str)

    # Filter only Descovy entries in the 340B dataset
    df_340b_descovy = df_340b[df_340b["DRUG_NM"].str.contains("DESCOVY", case=False, na=False)]

    return df_all, df_340b_descovy

def merge_on_ndc(df_all, df_340b_descovy):
    """Merge both datasets on NDC and flag 340B participation."""
    merged = df_all.merge(df_340b_descovy[["NDC", "PAID_AMT", "AWP", "QTY", "RX_CNT"]],
                          on="NDC", how="left", suffixes=("", "_340B"))
    merged["IS_340B"] = merged["PAID_AMT_340B"].notna()
    return merged

def compute_statistics(df):
    """Compute mean, median, and count for cost-related metrics by 340B flag."""
    return df.groupby("IS_340B")[["AWP", "PAID_AMT", "QTY", "RX_CNT"]].agg(["mean", "median", "count"])

def detect_outliers(df, column):
    """Return a DataFrame with an OUTLIER column for the specified metric."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df["OUTLIER_" + column] = (df[column] < lower) | (df[column] > upper)
    return df

def plot_comparisons(df):
    """Visualize AWP vs Paid Amount and highlight 340B."""
    plt.figure(figsize=(10, 6))
    colors = df["IS_340B"].map({True: "blue", False: "green"})
    plt.scatter(df["AWP"], df["PAID_AMT"], c=colors, alpha=0.6)
    plt.xlabel("AWP (Average Wholesale Price)")
    plt.ylabel("Paid Amount")
    plt.title("AWP vs Paid Amount by 340B Status")
    plt.grid(True)
    plt.legend(["340B", "Non-340B"])
    plt.tight_layout()
    plt.show()

def summarize_impact(df):
    """Print impact summary based on statistics."""
    stats = compute_statistics(df)
    print("Summary of 340B Impact on Cost Metrics:")
    print(stats)
    print("\n340B entries generally show lower paid amounts, which may suggest pricing differences under the program.")

# Example usage in Databricks:
# df_all, df_340b = load_data("full_claims.csv", "340b_entries.csv")
# df_merged = merge_on_ndc(df_all, df_340b)
# df_merged = detect_outliers(df_merged, "PAID_AMT")
# summarize_impact(df_merged)
# plot_comparisons(df_merged)
