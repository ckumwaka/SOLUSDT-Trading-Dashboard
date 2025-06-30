import pandas as pd
import numpy as np
import os

# Create sample historical data
dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='15T')
prices = np.cumprod(1 + np.random.normal(0.0001, 0.01, 1000))
df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': prices * 1.005,
    'low': prices * 0.995,
    'close': prices,
    'volume': np.random.randint(10000, 50000, 1000)
})

# Save to CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/historical_data.csv', index=False)
print("Sample data created at data/historical_data.csv")
