"""
Quick start example: How to use the forecasting module
"""

from core.forecast_loader import SignalLoader
from core.forecast_model import SignalForecaster
import pandas as pd


def example_1_list_signals():
    """Example 1: List all available signals"""
    print("=" * 70)
    print("EXAMPLE 1: List All Signals")
    print("=" * 70)
    
    loader = SignalLoader()
    signals = loader.load_all_signals()
    
    print(f"\nLoaded {len(signals)} signals\n")
    
    # Show first 10 signals
    for i, signal in enumerate(loader.list_signals()[:10], 1):
        meta = loader.get_metadata(signal)
        print(f"{i}. {signal}")
        print(f"   Points: {meta['n_points']} | Years: {meta['min_year']}-{meta['max_year']}\n")


def example_2_simple_forecast():
    """Example 2: Simple forecast"""
    print("=" * 70)
    print("EXAMPLE 2: Simple Forecast")
    print("=" * 70)
    
    loader = SignalLoader()
    signals = loader.load_all_signals()
    
    # Use first signal
    signal_name = list(signals.keys())[0]
    df = signals[signal_name]
    
    print(f"\nForecasting: {signal_name}\n")
    
    # Forecast
    forecaster = SignalForecaster()
    forecaster.fit(df, signal_name)
    predictions = forecaster.get_uncertainty_range(periods=5)
    
    print("Future Predictions (5 years):\n")
    print(predictions.to_string(index=False))


def example_3_search_and_forecast():
    """Example 3: Search for specific signal and forecast"""
    print("=" * 70)
    print("EXAMPLE 3: Search and Forecast")
    print("=" * 70)
    
    loader = SignalLoader()
    loader.load_all_signals()
    
    # Search for signals containing "Qualité"
    all_signals = loader.list_signals()
    matching = [s for s in all_signals if "Qualit" in s]
    
    if not matching:
        print("\nNo signals found matching 'Qualité'")
        return
    
    print(f"\nFound {len(matching)} signal(s) matching 'Qualité':\n")
    
    for i, signal in enumerate(matching, 1):
        print(f"{i}. {signal}")
    
    # Forecast the first matching signal
    signal_name = matching[0]
    print(f"\n\nForecasting: {signal_name}\n")
    
    df = loader.get_signal(signal_name)
    forecaster = SignalForecaster()
    forecaster.fit(df, signal_name)
    
    predictions = forecaster.get_uncertainty_range(periods=3)
    print("\nFuture Predictions (3 years):")
    print(predictions.to_string(index=False))
    
    # Calculate uncertainty statistics
    avg_uncertainty = (predictions['upper_bound'] - predictions['lower_bound']).mean()
    print(f"\nAverage uncertainty: ±{avg_uncertainty:.2f}")


def example_4_batch_forecast():
    """Example 4: Forecast multiple signals at once"""
    print("=" * 70)
    print("EXAMPLE 4: Batch Forecast")
    print("=" * 70)
    
    loader = SignalLoader()
    signals = loader.load_all_signals()
    
    # Select first 5 signals
    selected = dict(list(signals.items())[:5])
    
    print(f"\nForecasting {len(selected)} signals...\n")
    
    results = SignalForecaster.forecast_batch(selected, periods=3)
    
    for signal_name, prediction in results.items():
        if prediction is not None:
            future = prediction['future']
            print(f"\n{signal_name}")
            print(f"  2023: {future.iloc[0]['prediction']:.2f} (±{(future.iloc[0]['upper_bound'] - future.iloc[0]['lower_bound'])/2:.2f})")
            if len(future) > 1:
                print(f"  2024: {future.iloc[1]['prediction']:.2f} (±{(future.iloc[1]['upper_bound'] - future.iloc[1]['lower_bound'])/2:.2f})")
        else:
            print(f"\n{signal_name}: FAILED")


def example_5_export_csv():
    """Example 5: Export forecast to CSV"""
    print("=" * 70)
    print("EXAMPLE 5: Export to CSV")
    print("=" * 70)
    
    loader = SignalLoader()
    signals = loader.load_all_signals()
    
    signal_name = list(signals.keys())[0]
    df = signals[signal_name]
    
    forecaster = SignalForecaster()
    forecaster.fit(df, signal_name)
    predictions = forecaster.get_uncertainty_range(periods=5)
    
    # Prepare for export
    export_df = predictions.copy()
    export_df.columns = ['Annee', 'Valeur_Previsio', 'Limite_Inf_95', 'Limite_Sup_95', 'Prediction']
    
    # Save
    filename = "forecast_export.csv"
    export_df.to_csv(filename, index=False)
    
    print(f"\nForecast exported to: {filename}\n")
    print("Content preview:")
    print(export_df.head(10).to_string(index=False))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FORECASTING SYSTEM - QUICK START EXAMPLES")
    print("=" * 70 + "\n")
    
    # Run examples
    example_1_list_signals()
    input("\n[Press Enter to continue to Example 2]")
    
    example_2_simple_forecast()
    input("\n[Press Enter to continue to Example 3]")
    
    example_3_search_and_forecast()
    input("\n[Press Enter to continue to Example 4]")
    
    example_4_batch_forecast()
    input("\n[Press Enter to continue to Example 5]")
    
    example_5_export_csv()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
