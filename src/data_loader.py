# src/data_loader.py
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Import the direct constants from config.py
from src.config import PROJECT_ROOT, DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Robust data loader for the Xente transaction dataset.
    Uses DATA_PATH from config.py and automatically detects delimiter.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DATA_PATH
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        logger.info(f"DataLoader initialized with path: {self.data_path}")

    def _detect_delimiter(self) -> str:
        """Detect the correct delimiter by sampling the file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                sample = f.read(10240)  # Read first 10KB
        except UnicodeDecodeError:
            with open(self.data_path, 'r', encoding='latin-1') as f:
                sample = f.read(10240)

        delimiters = [",", "\t", ";", "|"]
        for delim in delimiters:
            if delim in sample:
                lines = sample.splitlines()[:5]
                col_counts = [len(line.split(delim)) for line in lines if line.strip()]
                if len(set(col_counts)) == 1 and max(col_counts) > 5:
                    logger.info(f"Detected delimiter: '{delim}'")
                    return delim
        logger.info("No delimiter detected confidently. Defaulting to comma.")
        return ","

    def load(self) -> pd.DataFrame:
        """Load the full dataset with safe parsing."""
        logger.info(f"Loading data from: {self.data_path}")
        delimiter = self._detect_delimiter()
        
        df = pd.read_csv(self.data_path, sep=delimiter, low_memory=False)
        logger.info(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

        # Safe datetime conversion
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
            invalid_count = df['TransactionStartTime'].isna().sum()
            if invalid_count > 0:
                logger.warning(f"{invalid_count} rows have invalid TransactionStartTime")

        # Ensure numeric columns
        numeric_cols = ['Amount', 'Value']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info("Data loading and basic preprocessing completed successfully.")
        return df

    def load_and_preview(self) -> pd.DataFrame:
        """Load and show a quick preview."""
        df = self.load()
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nDataset info:")
        df.info()
        return df

# Standalone test

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_and_preview()