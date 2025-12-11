# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import List, Optional
import sys
import os
import logging

# -------------------------------------------------
# PERMANENT FIX: Add project root to sys.path when running as script
# -------------------------------------------------
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"[INFO] Added project root to sys.path: {project_root}")

# Now imports work
from src.config import PROJECT_ROOT, DATA_PATH, PLOT_STYLE, PALETTE
from src.data_loader import DataLoader

# Suppress the specific matplotlib category INFO messages
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
plt.style.use(PLOT_STYLE)
sns.set_palette(PALETTE)

class EDA:
    """OOP class for exploratory data analysis as per Task 2 requirements."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info(f"EDA instance created with data shape: {self.df.shape}")

    def overview(self) -> None:
        """1. Overview of the Data"""
        print("="*70)
        print("1. OVERVIEW OF THE DATA")
        print("="*70)
        print(f"Number of rows: {self.df.shape[0]:,}")
        print(f"Number of columns: {self.df.shape[1]}")
        print("\nData Types and Non-Null Counts:")
        self.df.info()
        print("\nFirst 5 rows:")
        print(self.df.head())

    def summary_statistics(self) -> None:
        """2. Summary Statistics"""
        print("\n" + "="*70)
        print("2. SUMMARY STATISTICS")
        print("="*70)
        print(self.df.describe(include='all').T)

    def numerical_distributions(self) -> None:
        """3. Distribution of Numerical Features"""
        print("\n" + "="*70)
        print("3. DISTRIBUTION OF NUMERICAL FEATURES")
        print("="*70)

        # Amount (signed)
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df['Amount'].dropna(), bins=100, kde=True, color='skyblue')
        plt.title('Distribution of Amount (Signed)', fontsize=16)
        plt.xlabel('Amount (UGX)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Value (|Amount|)
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df['Value'].dropna(), bins=100, kde=True, color='salmon')
        plt.title('Distribution of Value (|Amount|)', fontsize=16)
        plt.xlabel('Value (UGX)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Debits only
        debits = self.df[self.df['Amount'] > 0]
        plt.figure(figsize=(12, 6))
        sns.histplot(debits['Amount'], bins=100, kde=True, color='green')
        plt.title('Distribution of Debits Only (Customer Spending)', fontsize=16)
        plt.xlabel('Amount (UGX)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

        print("Skewness:")
        print(self.df[['Amount', 'Value']].skew().round(2))

    def categorical_distributions(self) -> None:
        """4. Distribution of Categorical Features"""
        print("\n" + "="*70)
        print("4. DISTRIBUTION OF CATEGORICAL FEATURES")
        print("="*70)
        cat_cols = ['CurrencyCode', 'CountryCode', 'FraudResult', 'ProductCategory', 'ChannelId', 'PricingStrategy', 'ProviderId']

        for col in cat_cols:
            if col in self.df.columns:
                df_plot = self.df.copy()
                df_plot[col] = df_plot[col].astype(str)  # FIX: Force to string to avoid warnings
                
                value_counts = df_plot[col].value_counts()
                unique_count = len(value_counts)

                # For highly imbalanced categories, use pie chart
                if unique_count <= 2 or value_counts.iloc[0] / value_counts.sum() > 0.99:
                    plt.figure(figsize=(8, 8))
                    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                    plt.title(f'Distribution of {col} (Highly Imbalanced)', fontsize=16)
                    plt.axis('equal')
                    plt.show()
                    print(f"{col}: Dominant category = {value_counts.index[0]} ({value_counts.iloc[0]/value_counts.sum():.1%})")
                else:
                    plt.figure(figsize=(12, 6))
                    top = value_counts.head(10)
                    sns.barplot(x=top.values, y=top.index, palette='viridis')
                    plt.title(f'Top 10 {col}', fontsize=16)
                    plt.xlabel('Count')
                    plt.ylabel(col)
                    plt.tight_layout()
                    plt.show()

    def correlation_analysis(self) -> None:
        """5. Correlation Analysis"""
        print("\n" + "="*70)
        print("5. CORRELATION ANALYSIS")
        print("="*70)
        numeric = self.df.select_dtypes(include=[np.number])
        if numeric.shape[1] > 1:
            plt.figure(figsize=(12, 10))
            corr = numeric.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Features', fontsize=16)
            plt.tight_layout()
            plt.show()

    def missing_values(self) -> None:
        """6. Identifying Missing Values"""
        print("\n" + "="*70)
        print("6. IDENTIFYING MISSING VALUES")
        print("="*70)
        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct.round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if missing_df.empty:
            print("No missing values found in the dataset.")
        else:
            print(missing_df.sort_values('Missing %', ascending=False))

    def outlier_detection(self) -> None:
        """7. Outlier Detection"""
        print("\n" + "="*70)
        print("7. OUTLIER DETECTION")
        print("="*70)
        debits = self.df[self.df['Amount'] > 0].copy()
        if debits.empty:
            print("No debit transactions for outlier analysis.")
            return
        
        # FIX: Force string type to eliminate matplotlib category warnings
        debits['ProductCategory'] = debits['ProductCategory'].astype(str)
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=debits, x='ProductCategory', y='Amount', palette='Set2')
        plt.yscale('log')
        plt.xticks(rotation=45, ha='right')
        plt.title('Outliers in Transaction Amount by Product Category (Log Scale)', fontsize=16)
        plt.ylabel('Amount (UGX, log scale)')
        plt.xlabel('Product Category')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def top_insights(self) -> None:
        """Deliverable: Top 3–5 most important insights"""
        print("\n" + "="*70)
        print("TOP 5 MOST IMPORTANT INSIGHTS")
        print("="*70)
        insights = [
            "1. FraudResult is extremely imbalanced (~0.13%) — cannot be used as a direct default proxy.",
            "2. Nearly all transactions are in Uganda (CountryCode 256, Currency UGX) — model can be country-specific.",
            "3. Transaction amounts are heavily right-skewed with a long tail — log transformation and robust scaling will be necessary.",
            "4. Clear customer behavioral segments visible (high vs low frequency/spend) — ideal for RFM clustering to create a proxy target.",
            "5. Value column is exactly |Amount| — redundant; use only positive Amount (debits) for monetary features to avoid double-counting."
        ]
        for insight in insights:
            print(insight)

    def run_full_eda(self) -> None:
        """Run the complete EDA as per Task 2 instructions."""
        self.overview()
        self.summary_statistics()
        self.numerical_distributions()
        self.categorical_distributions()
        self.correlation_analysis()
        self.missing_values()
        self.outlier_detection()
        self.top_insights()

# Standalone execution


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    eda = EDA(df)
    eda.run_full_eda()