"""
Integration module for forecasting system into main application.
This module bridges the forecasting engine with your existing app.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from forecast_loader import SignalLoader
from forecast_model import SignalForecaster


class ForecastingModule:
    """Forecasting module for integration into main app."""
    
    def __init__(self):
        self.loader = None
        self.signals = None
        self.forecaster = None
        
    @st.cache_resource
    def _initialize(self):
        """Initialize loader and load all signals."""
        loader = SignalLoader()
        signals = loader.load_all_signals()
        return loader, signals
    
    def get_available_signals(self):
        """Get list of available signal names."""
        if self.loader is None:
            self.loader, self.signals = self._initialize()
        return self.loader.list_signals()
    
    def forecast_signal(self, signal_name: str, periods: int = 5):
        """
        Forecast a specific signal.
        
        Args:
            signal_name: Name of the signal
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        if self.loader is None:
            self.loader, self.signals = self._initialize()
        
        try:
            # Get signal
            df = self.loader.get_signal(signal_name)
            meta = self.loader.get_metadata(signal_name)
            
            # Forecast
            forecaster = SignalForecaster()
            forecaster.fit(df, signal_name)
            predictions = forecaster.get_uncertainty_range(periods)
            
            return {
                'success': True,
                'signal_name': signal_name,
                'metadata': meta,
                'predictions': predictions,
                'data': df
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_forecast_plot(self, signal_name: str, periods: int = 5):
        """
        Create interactive forecast plot.
        
        Args:
            signal_name: Signal to forecast
            periods: Number of periods
            
        Returns:
            Plotly figure
        """
        result = self.forecast_signal(signal_name, periods)
        
        if not result['success']:
            st.error(f"Forecast failed: {result.get('error', 'Unknown error')}")
            return None
        
        predictions = result['predictions']
        
        # Create figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=predictions.iloc[:-periods]['year'],
            y=predictions.iloc[:-periods]['prediction'],
            name='Donnees Historiques',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2.5),
            marker=dict(size=6)
        ))
        
        # Forecast
        forecast_data = predictions.iloc[-periods:]
        
        fig.add_trace(go.Scatter(
            x=forecast_data['year'],
            y=forecast_data['prediction'],
            name='Prevision',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2.5, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Uncertainty bands
        fig.add_trace(go.Scatter(
            x=forecast_data['year'],
            y=forecast_data['upper_bound'],
            name='Limite Superieure',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['year'],
            y=forecast_data['lower_bound'],
            name='Limite Inferieure',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(44, 160, 44, 0.2)',
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f"Prevision: {signal_name}",
            xaxis_title="Annee",
            yaxis_title="Valeur",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def export_forecast_csv(self, signal_name: str, periods: int = 5):
        """
        Export forecast to CSV.
        
        Args:
            signal_name: Signal to forecast
            periods: Number of periods
            
        Returns:
            CSV string
        """
        result = self.forecast_signal(signal_name, periods)
        
        if not result['success']:
            return None
        
        predictions = result['predictions']
        predictions.columns = ['Annee', 'Valeur', 'Limite_Inf', 'Limite_Sup', 'Prediction']
        
        return predictions.to_csv(index=False)


def main():
    """Demo usage of the forecasting module."""
    st.set_page_config(page_title="Forecasting Module", layout="wide")
    
    st.title("Forecasting Module Integration Demo")
    st.markdown("This shows how to integrate the forecasting module into your app.")
    
    # Initialize module
    forecast_module = ForecastingModule()
    
    # Get signals
    signals = forecast_module.get_available_signals()
    
    st.sidebar.header("Configuration")
    selected_signal = st.sidebar.selectbox("Select signal:", signals)
    periods = st.sidebar.slider("Forecast periods:", 1, 15, 5)
    
    # Forecast
    result = forecast_module.forecast_signal(selected_signal, periods)
    
    if result['success']:
        st.success(f"Forecast generated for: {result['signal_name']}")
        
        # Show plot
        fig = forecast_module.create_forecast_plot(selected_signal, periods)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data
        st.subheader("Forecast Data")
        st.dataframe(result['predictions'], use_container_width=True)
        
        # Download
        csv = forecast_module.export_forecast_csv(selected_signal, periods)
        st.download_button(
            "Download CSV",
            csv,
            f"forecast_{selected_signal.replace(' ', '_')}.csv"
        )
    else:
        st.error(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
