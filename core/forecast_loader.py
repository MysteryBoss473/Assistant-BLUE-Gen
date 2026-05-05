"""
Data loader for time series signals from Excel files.
Extracts all signals from multiple sheets and files for forecasting.
"""

import os
import pandas as pd
import openpyxl
from typing import Dict, List, Tuple
from pathlib import Path


class SignalLoader:
    """Load and extract all individual signals from Excel dataset."""
    
    def __init__(self, dataset_dir: str = "dataset/signaux"):
        self.dataset_dir = dataset_dir
        self.signals_db: Dict[str, pd.Series] = {}
        self.metadata: Dict[str, dict] = {}
        
    def load_all_signals(self) -> Dict[str, pd.Series]:
        """
        Load all signals from all Excel files in dataset.
        Returns: Dictionary with signal_name -> (years, values) Series
        """
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        excel_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.xlsx')]
        
        for file in sorted(excel_files):
            filepath = os.path.join(self.dataset_dir, file)
            self._load_file(filepath)
        
        return self.signals_db
    
    def _load_file(self, filepath: str) -> None:
        """Load all signals from a single Excel file."""
        filename = os.path.basename(filepath)
        print(f"Loading {filename}...")
        
        wb = openpyxl.load_workbook(filepath)
        
        for sheet_name in wb.sheetnames:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            # Fix column name for year column
            year_col = self._find_year_column(df)
            if year_col is None:
                print(f"  [SKIP] Sheet '{sheet_name}': No year column found")
                continue
            
            # Extract signals from this sheet
            self._extract_signals_from_sheet(df, year_col, filename, sheet_name)
    
    def _find_year_column(self, df: pd.DataFrame) -> str:
        """Find the year column (Annee, Rubrique Name, etc.)."""
        year_candidates = ['Annee', 'annee', 'Year', 'year', 'Rubrique Name']
        for col in year_candidates:
            if col in df.columns:
                return col
        
        # If not found, check first column
        first_col = df.columns[0]
        if pd.api.types.is_integer_dtype(df[first_col]):
            return first_col
        
        return None
    
    def _extract_signals_from_sheet(self, df: pd.DataFrame, year_col: str, 
                                   filename: str, sheet_name: str) -> None:
        """Extract individual signals from a sheet."""
        # Get years as index
        years = df[year_col].values
        
        # Skip the year column and extract all other columns as signals
        signal_columns = [col for col in df.columns if col != year_col]
        
        for col in signal_columns:
            values = df[col].values
            
            # Skip if all NaN or non-numeric
            if pd.isna(values).all():
                continue
            
            # Create unique signal name
            prefix = filename.replace('.xlsx', '')
            if sheet_name != 'ObservationData':
                signal_name = f"{prefix} - {sheet_name} - {col}"
            else:
                signal_name = f"{prefix} - {col}"
            
            # Create series (remove NaN values, keep matching years)
            valid_mask = ~pd.isna(values)
            valid_years = years[valid_mask]
            valid_values = values[valid_mask]
            
            if len(valid_values) > 3:  # At least 3 points for forecasting
                # Create DataFrame for Prophet format
                prophet_df = pd.DataFrame({
                    'ds': pd.to_datetime(valid_years, format='%Y'),
                    'y': valid_values
                })
                
                self.signals_db[signal_name] = prophet_df
                self.metadata[signal_name] = {
                    'file': filename,
                    'sheet': sheet_name,
                    'column': col,
                    'n_points': len(valid_values),
                    'min_year': int(valid_years.min()),
                    'max_year': int(valid_years.max())
                }
                
                print(f"  [OK] {signal_name} ({len(valid_values)} points)")
    
    def get_signal(self, signal_name: str) -> pd.DataFrame:
        """Get a specific signal as DataFrame for Prophet."""
        if signal_name not in self.signals_db:
            raise ValueError(f"Signal not found: {signal_name}")
        return self.signals_db[signal_name]
    
    def list_signals(self) -> List[str]:
        """Get list of all available signals."""
        return sorted(list(self.signals_db.keys()))
    
    def get_metadata(self, signal_name: str) -> dict:
        """Get metadata for a signal."""
        return self.metadata.get(signal_name, {})
    
    def get_all_metadata(self) -> Dict[str, dict]:
        """Get metadata for all signals."""
        return self.metadata


if __name__ == "__main__":
    # Test the loader
    loader = SignalLoader()
    signals = loader.load_all_signals()
    print(f"\n[SUCCESS] Loaded {len(signals)} signals")
    
    print("\n" + "="*70)
    print("Available signals:")
    print("="*70)
    for i, signal in enumerate(loader.list_signals(), 1):
        meta = loader.get_metadata(signal)
        print(f"{i}. {signal}")
        print(f"   Points: {meta.get('n_points')} | Years: {meta.get('min_year')}-{meta.get('max_year')}")
