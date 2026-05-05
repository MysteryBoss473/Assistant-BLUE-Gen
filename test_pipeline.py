#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from core.signal_forecast import SignalForecastPipeline

# Test basic initialization
pipeline = SignalForecastPipeline()
print('[OK] Pipeline initialized successfully')

# Try to load signals
result = pipeline.ingest_directory('dataset/signaux')
print('[OK] Signals loaded:', result.get('signals_loaded', 0), 'signals')
print('[OK] Available:', len(pipeline.available_signals), 'signals')

# Test a simple forecast
if pipeline.available_signals:
    signal_key = pipeline.available_signals[0]
    print('\n[TEST] Forecasting signal:', signal_key)
    run_result = pipeline.run(signal_key=signal_key, periods=3)
    print('[OK] Forecast result status:', run_result.get('status'))
    if run_result.get('status') == 'success':
        forecast = run_result.get('forecast', {})
        print('[OK] Forecast model:', forecast.get('model'))
        print('[OK] Periods:', forecast.get('periods'))
        print('[OK] Confidence level:', forecast.get('confidence_level'))
        print('[OK] Number of forecast points:', len(forecast.get('forecast', [])))
