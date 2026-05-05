#!/usr/bin/env python
"""
CLI utility for signal forecasting.
Allows forecasting without Streamlit interface.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from forecast_loader import SignalLoader
from forecast_model import SignalForecaster


def main():
    parser = argparse.ArgumentParser(
        description="Forecast time series signals with uncertainty bands"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List signals command
    list_parser = subparsers.add_parser('list', help='List all available signals')
    list_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed info')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Forecast a signal')
    forecast_parser.add_argument('signal', type=str, help='Signal name (use "list" to see options)')
    forecast_parser.add_argument('-p', '--periods', type=int, default=5, help='Forecast periods (default: 5)')
    forecast_parser.add_argument('-o', '--output', type=str, help='Output CSV file')
    forecast_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for signals by keyword')
    search_parser.add_argument('keyword', type=str, help='Search keyword')
    
    args = parser.parse_args()
    
    # Load signals
    try:
        loader = SignalLoader()
        signals = loader.load_all_signals()
    except Exception as e:
        print(f"[ERROR] Failed to load signals: {e}")
        return 1
    
    if args.command == 'list':
        print("\n" + "="*80)
        print("AVAILABLE SIGNALS")
        print("="*80)
        
        signal_list = loader.list_signals()
        if not signal_list:
            print("No signals found!")
            return 1
        
        for i, signal in enumerate(signal_list, 1):
            if args.verbose:
                meta = loader.get_metadata(signal)
                print(f"\n{i}. {signal}")
                print(f"   File: {meta.get('file')}")
                print(f"   Sheet: {meta.get('sheet')}")
                print(f"   Points: {meta.get('n_points')}")
                print(f"   Period: {meta.get('min_year')}-{meta.get('max_year')}")
            else:
                print(f"{i}. {signal}")
        
        print("\n" + "="*80)
        print(f"Total: {len(signal_list)} signals")
        print("="*80 + "\n")
        
    elif args.command == 'forecast':
        signal_name = args.signal
        
        # Find matching signal
        all_signals = loader.list_signals()
        matching = [s for s in all_signals if signal_name.lower() in s.lower()]
        
        if not matching:
            print(f"\n[ERROR] Signal not found: {signal_name}")
            print("\nDid you mean:")
            for s in all_signals[:5]:
                print(f"  - {s}")
            return 1
        
        if len(matching) > 1:
            print(f"\n[WARNING] Multiple matches found. Using first match.")
        
        signal_name = matching[0]
        
        print(f"\n{'='*80}")
        print(f"FORECASTING: {signal_name}")
        print(f"{'='*80}")
        
        try:
            # Get signal and fit model
            signal_df = loader.get_signal(signal_name)
            forecaster = SignalForecaster()
            forecaster.fit(signal_df, signal_name)
            
            # Get predictions
            print(f"\n[*] Generating {args.periods} period(s) forecast...")
            predictions = forecaster.get_uncertainty_range(args.periods)
            
            print("\nFuture Predictions:")
            print(predictions.to_string(index=False))
            
            # Save to file if requested
            if args.output:
                predictions.to_csv(args.output, index=False)
                print(f"\n[OK] Results saved to: {args.output}")
            
        except Exception as e:
            print(f"\n[ERROR] Forecast failed: {e}")
            return 1
    
    elif args.command == 'search':
        keyword = args.keyword.lower()
        all_signals = loader.list_signals()
        matches = [s for s in all_signals if keyword in s.lower()]
        
        if not matches:
            print(f"[INFO] No signals matching '{keyword}' found")
            return 1
        
        print(f"\nSignals matching '{keyword}':")
        for i, signal in enumerate(matches, 1):
            meta = loader.get_metadata(signal)
            print(f"{i}. {signal}")
            print(f"   Points: {meta.get('n_points')} | Years: {meta.get('min_year')}-{meta.get('max_year')}")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
