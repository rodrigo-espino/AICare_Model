import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

import joblib

dataset_preprocessed = '../data/processed/diabetes.csv'
pd_diabetes = pd.read_csv(dataset_preprocessed)

X = pd_diabetes.drop('Diabetes', axis=1)
y = pd_diabetes.Diabetes

def create_mlp_model():
    model = Sequential()
    model.add(Dense(45, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)


pipeline = make_pipeline(
    StandardScaler(),
    KerasClassifier(model=create_mlp_model, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
)


pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'mlp_pipeline.pkl')

