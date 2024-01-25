import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime

# Load the generated dataset
df = pd.read_csv("dataset.csv")

# Convert timestamp to numerical values (in seconds)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp_seconds'] = (df['timestamp'] - datetime(1970, 1, 1)).dt.total_seconds()

# Features (X) and target variable (y)
X = df[['timestamp_seconds', 'gate_allocated']]
y = df['wait_time']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict wait time for a new entry
new_entry = pd.DataFrame({'timestamp_seconds': [datetime.now().timestamp()], 'gate_allocated': [np.random.randint(1, 5)]})
predicted_wait_time = model.predict(new_entry)

print(f"Predicted Wait Time for New Entry: {predicted_wait_time[0]:.2f} seconds")
