import joblib
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_mlp_model():
    model = Sequential()
    model.add(Dense(64, input_dim=11, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_file = 'src/mlp_pipeline.pkl'
loaded_model = joblib.load(model_file)

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    features = [
        data['Cholesterol'],
        data['Glucose'],
        data['HDL Chol'],
        data['Chol/HDL ratio'],
        data['Age'],
        data['Gender'],
        data['Height'],
        data['Weight'],
        data['BMI'],
        data['Systolic BP'],
        data['Diastolic BP'],
        data['waist'],
        data['hip'],
        data['Waist/hip ratio']
    ]

    input_data = np.array([features])

    prediction = loaded_model.predict(input_data)


    return jsonify({
        'prediction': int(prediction[0])
    })

if __name__ == '__main__':
    app.run(debug=True)