"""
Time series forecasting with Prophet.
Generates predictions with uncertainty intervals (confidence bands).
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SignalForecaster:
    """Forecast signals with uncertainty intervals using Prophet."""
    
    def __init__(self, yearly_seasonality: bool = False):
        """
        Initialize forecaster.
        
        Args:
            yearly_seasonality: Whether to model yearly seasonality
        """
        self.yearly_seasonality = yearly_seasonality
        self.model = None
        self.forecast = None
        self.train_data = None
        
    def fit(self, df: pd.DataFrame, signal_name: str = "signal") -> None:
        """
        Fit Prophet model to time series data.
        
        Args:
            df: DataFrame with columns 'ds' (datetime) and 'y' (values)
            signal_name: Name of the signal (for logging)
        """
        self.train_data = df.copy()
        
        # Initialize Prophet with appropriate settings for yearly data
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=False,
            weekly_seasonality=False,
            interval_width=0.95,  # 95% confidence interval
            seasonality_mode='additive',
        )
        
        # Fit the model
        print(f"Training forecaster for: {signal_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)
        print("[OK] Model trained successfully")
    
    def predict(self, periods: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for specified periods ahead.
        
        Args:
            periods: Number of years to forecast
            
        Returns:
            Dictionary with:
            - 'forecast': Full forecast DataFrame
            - 'plot_data': Simplified DataFrame for plotting
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='YS')
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.forecast = self.model.predict(future)
        
        # Prepare plot data with uncertainty bands
        plot_data = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        plot_data['year'] = plot_data['ds'].dt.year
        plot_data = plot_data.rename(columns={
            'yhat': 'prediction',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        })
        
        return {
            'forecast': self.forecast,
            'plot_data': plot_data
        }
    
    def get_prediction_with_bounds(self, periods: int = 5) -> pd.DataFrame:
        """
        Get predictions with uncertainty bounds for easy visualization.
        
        Returns:
            DataFrame with columns: year, prediction, lower_bound, upper_bound
        """
        prediction_dict = self.predict(periods)
        plot_data = prediction_dict['plot_data']
        
        # Split into historical and future
        n_train = len(self.train_data)
        historical = plot_data.iloc[:n_train].copy()
        future = plot_data.iloc[n_train:].copy()
        
        return {
            'historical': historical,
            'future': future
        }
    
    def get_uncertainty_range(self, periods: int = 5) -> pd.DataFrame:
        """
        Get the uncertainty envelope (min and max bounds) for visualization.
        
        Returns:
            DataFrame with year, lower_bound, upper_bound, mid_point
        """
        prediction_dict = self.predict(periods)
        plot_data = prediction_dict['plot_data'].copy()
        
        n_train = len(self.train_data)
        future_forecast = plot_data.iloc[n_train:].copy()
        
        future_forecast['mid_point'] = (
            (future_forecast['lower_bound'] + future_forecast['upper_bound']) / 2
        )
        
        return future_forecast[['year', 'lower_bound', 'upper_bound', 'mid_point', 'prediction']]
    
    @staticmethod
    def forecast_batch(signals_dict: Dict[str, pd.DataFrame], 
                      periods: int = 5) -> Dict[str, Dict]:
        """
        Forecast multiple signals at once.
        
        Args:
            signals_dict: Dictionary of signal_name -> DataFrame (with 'ds', 'y' columns)
            periods: Number of periods to forecast
            
        Returns:
            Dictionary of signal_name -> forecast results
        """
        results = {}
        
        for signal_name, df in signals_dict.items():
            try:
                forecaster = SignalForecaster()
                forecaster.fit(df, signal_name)
                results[signal_name] = forecaster.get_prediction_with_bounds(periods)
                print(f"✓ {signal_name}")
            except Exception as e:
                print(f"✗ {signal_name}: {str(e)}")
                results[signal_name] = None
        
        return results


if __name__ == "__main__":
    # Test example
    from forecast_loader import SignalLoader
    
    loader = SignalLoader()
    signals = loader.load_all_signals()
    
    # Test with first signal
    first_signal = list(signals.items())[0]
    signal_name, df = first_signal
    
    print(f"\n{'='*70}")
    print(f"Testing forecast for: {signal_name}")
    print(f"{'='*70}")
    
    forecaster = SignalForecaster()
    forecaster.fit(df, signal_name)
    
    # Get predictions
    predictions = forecaster.get_uncertainty_range(periods=5)
    print("\nFuture Predictions (5 years):")
    print(predictions)
