import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv("output/landmarks_train.csv")

# X = image paths, Y = landmark coordinates
Y = df.drop(columns=["image_path"]).values
scaler = MinMaxScaler()
Y = scaler.fit_transform(Y)

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(Y, Y, test_size=0.2, random_state=42)

# Simple model (autoencoder-style regression)
model = models.Sequential([
    layers.Input(shape=(Y.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(Y.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=16)
model.save("output/facial_landmark_model.h5")

print("✅ Model saved as facial_landmark_model.h5")
