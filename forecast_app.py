"""
Streamlit app for interactive time series forecasting with uncertainty visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from forecast_loader import SignalLoader
from forecast_model import SignalForecaster


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Signal Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .forecast-header {
        color: #1f77b4;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Initialize Session State & Load Data
# ============================================================================
@st.cache_resource
def load_signals():
    """Load all signals from Excel files."""
    loader = SignalLoader()
    signals = loader.load_all_signals()
    return loader, signals


def get_plot_color_scheme():
    """Return color scheme for plots."""
    return {
        'historical': '#1f77b4',      # Blue
        'prediction': '#ff7f0e',      # Orange  
        'upper_bound': '#2ca02c',     # Green (light)
        'lower_bound': '#2ca02c',     # Green (light)
        'uncertainty': 'rgba(44, 160, 44, 0.2)'  # Light green with transparency
    }


# ============================================================================
# Main App
# ============================================================================
def main():
    st.title("📈 Système de Prévision de Signaux Temporels")
    st.markdown("*Forecasting avec Enveloppes d'Incertitude*")
    
    # Load data
    loader, signals = load_signals()
    signal_names = loader.list_signals()
    
    if not signal_names:
        st.error("❌ Aucun signal trouvé dans le dataset!")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Signal selection
    selected_signal = st.sidebar.selectbox(
        "Sélectionnez un signal à prévoir:",
        signal_names,
        key="signal_selector"
    )
    
    # Forecast parameters
    st.sidebar.subheader("Paramètres de Prévision")
    forecast_periods = st.sidebar.slider(
        "Périodes à prévoir (années):",
        min_value=1,
        max_value=15,
        value=5,
        step=1
    )
    
    # Show metadata
    meta = loader.get_metadata(selected_signal)
    st.sidebar.subheader("📊 Métadonnées du Signal")
    st.sidebar.info(f"""
    **Fichier:** {meta.get('file', 'N/A')}
    
    **Feuille:** {meta.get('sheet', 'N/A')}
    
    **Points de données:** {meta.get('n_points', 'N/A')}
    
    **Période couverte:** {meta.get('min_year', 'N/A')} - {meta.get('max_year', 'N/A')}
    """)
    
    # ========================================================================
    # Main Content
    # ========================================================================
    st.markdown(f"### 📌 Signal Sélectionné")
    st.markdown(f"**{selected_signal}**")
    
    # Get signal data
    signal_df = loader.get_signal(selected_signal)
    
    try:
        # Fit forecaster
        forecaster = SignalForecaster(yearly_seasonality=False)
        forecaster.fit(signal_df, selected_signal)
        
        # Get predictions
        predictions = forecaster.get_prediction_with_bounds(forecast_periods)
        historical = predictions['historical']
        future = predictions['future']
        
        # ====================================================================
        # Statistics Cards
        # ====================================================================
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_val = signal_df['y'].min()
            st.metric("📉 Valeur Min", f"{min_val:.2f}")
        
        with col2:
            max_val = signal_df['y'].max()
            st.metric("📈 Valeur Max", f"{max_val:.2f}")
        
        with col3:
            mean_val = signal_df['y'].mean()
            st.metric("📊 Moyenne", f"{mean_val:.2f}")
        
        with col4:
            last_val = signal_df['y'].iloc[-1]
            st.metric("🎯 Dernière Valeur", f"{last_val:.2f}")
        
        # ====================================================================
        # Visualization: Full Forecast with Uncertainty Bands
        # ====================================================================
        st.markdown("### 📈 Prévision avec Enveloppe d'Incertitude")
        
        colors = get_plot_color_scheme()
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['year'],
            y=historical['prediction'],
            name='Données Historiques',
            mode='lines+markers',
            line=dict(color=colors['historical'], width=2.5),
            marker=dict(size=6),
            hovertemplate='<b>Année:</b> %{x}<br><b>Valeur:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=future['year'],
            y=future['prediction'],
            name='Prévision',
            mode='lines+markers',
            line=dict(color=colors['prediction'], width=2.5, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Année:</b> %{x}<br><b>Prévision:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Uncertainty upper bound
        fig.add_trace(go.Scatter(
            x=future['year'],
            y=future['upper_bound'],
            name='Limite Supérieure (95%)',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip'
        ))
        
        # Uncertainty lower bound
        fig.add_trace(go.Scatter(
            x=future['year'],
            y=future['lower_bound'],
            name='Limite Inférieure (95%)',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor=colors['uncertainty'],
            fill='tonexty',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Prévision: {selected_signal}",
            xaxis_title="Année",
            yaxis_title="Valeur",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ====================================================================
        # Table: Future Predictions Detail
        # ====================================================================
        st.markdown("### 📋 Détails des Prévisions Futures")
        
        future_table = future[['year', 'lower_bound', 'prediction', 'upper_bound']].copy()
        future_table.columns = ['Année', 'Limite Inf. (95%)', 'Prévision', 'Limite Sup. (95%)']
        future_table['Année'] = future_table['Année'].astype(int)
        
        # Format numbers
        for col in ['Limite Inf. (95%)', 'Prévision', 'Limite Sup. (95%)']:
            future_table[col] = future_table[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(future_table, use_container_width=True, hide_index=True)
        
        # ====================================================================
        # Uncertainty Statistics
        # ====================================================================
        st.markdown("### 📊 Statistiques d'Incertitude")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_uncertainty = (future['upper_bound'] - future['lower_bound']).mean()
            st.metric("Incertitude Moyenne", f"±{avg_uncertainty:.2f}")
        
        with col2:
            max_uncertainty = (future['upper_bound'] - future['lower_bound']).max()
            st.metric("Incertitude Max", f"±{max_uncertainty:.2f}")
        
        with col3:
            min_uncertainty = (future['upper_bound'] - future['lower_bound']).min()
            st.metric("Incertitude Min", f"±{min_uncertainty:.2f}")
        
        # ====================================================================
        # Download Data
        # ====================================================================
        st.markdown("### 💾 Exporter les Résultats")
        
        export_df = pd.concat([
            historical[['year', 'prediction', 'lower_bound', 'upper_bound']],
            future[['year', 'prediction', 'lower_bound', 'upper_bound']]
        ], ignore_index=True)
        
        export_df.columns = ['Année', 'Valeur', 'Limite_Inf', 'Limite_Sup']
        
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"forecast_{selected_signal.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la prévision: {str(e)}")
        st.info("Cet signal peut ne pas avoir assez de points de données pour une bonne prévision.")
    
    # ========================================================================
    # Footer
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    **À propos:** Ce système utilise Prophet (Facebook) pour générer des prévisions
    avec des intervalles de confiance à 95%. Les enveloppes d'incertitude augmentent
    avec l'horizon de prévision, reflétant l'incertitude croissante.
    """)


if __name__ == "__main__":
    main()
