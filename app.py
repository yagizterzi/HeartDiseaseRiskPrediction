from flask import Flask, request, render_template_string, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("models/best_rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")  # Load the preprocessor used during training

feature_names = [
    'age','sex' ,'resting_blood_pressure', 'cholestrol', 'fasting_blood_sugar', 'max_heart_rate',
    'st_depression_exercise', 'major_vessels', 'chest_pain_type', 'resting_electrocardiograh',
    'exercise_induced_angina', 'peak_exercise_slope', 'thalassemia'
]

# Define categorical columns and their acceptable values based on training
categorical_cols = ['chest_pain_type', 'resting_electrocardiograh', 
                   'exercise_induced_angina', 'peak_exercise_slope', 
                   'thalassemia', 'sex', 'fasting_blood_sugar', 'major_vessels']

# Route for displaying the form
@app.route('/')
def home():
    return render_template_string('''
    <h1>Heart Disease Risk Prediction</h1>
    <p>Please enter your health information below. For categorical fields, use the specified codes.</p>
    
    <form method="POST" action="/predict">
        <label for="age">Age: </label>
        <input type="number" name="age" min="20" max="100" required><br><br>
        
        <label for="sex">Sex: </label>
        <select name="sex" required>
            <option value="0">Female (0)</option>
            <option value="1">Male (1)</option>
        </select><br><br>   
        
        <label for="resting_blood_pressure">Resting Blood Pressure (mmHg): </label>
        <input type="number" name="resting_blood_pressure" min="80" max="200" required><br><br>
        
        <label for="cholestrol">Cholesterol (mg/dl): </label>
        <input type="number" name="cholestrol" min="100" max="600" required><br><br>
        
        <label for="fasting_blood_sugar">Fasting Blood Sugar: </label>
        <select name="fasting_blood_sugar" required>
            <option value="0">Less than 120 mg/dl (0)</option>
            <option value="1">Greater than 120 mg/dl (1)</option>
        </select><br><br>
        
        <label for="max_heart_rate">Max Heart Rate (bpm): </label>
        <input type="number" name="max_heart_rate" min="60" max="220" required><br><br>
        
        <label for="st_depression_exercise">ST Depression Induced by Exercise: </label>
        <input type="number" step="0.1" name="st_depression_exercise" min="0" max="10" required><br><br>
        
        <label for="major_vessels">Number of Major Vessels (0-3): </label>
        <select name="major_vessels" required>
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
        </select><br><br>
        
        <label for="chest_pain_type">Chest Pain Type: </label>
        <select name="chest_pain_type" required>
            <option value="0">Typical Angina (0)</option>
            <option value="1">Atypical Angina (1)</option>
            <option value="2">Non-anginal Pain (2)</option>
            <option value="3">Asymptomatic (3)</option>
        </select><br><br>
        
        <label for="resting_electrocardiograh">Resting Electrocardiographic Results: </label>
        <select name="resting_electrocardiograh" required>
            <option value="0">Normal (0)</option>
            <option value="1">ST-T Wave Abnormality (1)</option>
            <option value="2">Left Ventricular Hypertrophy (2)</option>
        </select><br><br>
        
        <label for="exercise_induced_angina">Exercise Induced Angina: </label>
        <select name="exercise_induced_angina" required>
            <option value="0">No (0)</option>
            <option value="1">Yes (1)</option>
        </select><br><br>
        
        <label for="peak_exercise_slope">Peak Exercise ST Segment Slope: </label>
        <select name="peak_exercise_slope" required>
            <option value="1">Upsloping (1)</option>
            <option value="2">Flat (2)</option>
            <option value="3">Downsloping (3)</option>
        </select><br><br>
        
        <label for="thalassemia">Thalassemia: </label>
        <select name="thalassemia" required>
            <option value="3">Normal (3)</option>
            <option value="6">Fixed Defect (6)</option>
            <option value="7">Reversible Defect (7)</option>
        </select><br><br>
        
        <input type="submit" value="Predict">
    </form>
    ''')

# Route for prediction
# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            # Convert to proper type - ensure categorical values are integers
            if feature in categorical_cols:
                form_data[feature] = int(value)
            else:
                form_data[feature] = float(value)
        
        # Create DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Debug info
        print("Input data:")
        print(input_df)
        print("Data types:")
        print(input_df.dtypes)
        
        # Transform features using the preprocessor
        features_processed = preprocessor.transform(input_df)
        print(f"Processed features shape: {features_processed.shape}")
        
        # Scale the features
        scaled_features = scaler.transform(features_processed)
        print(f"Scaled features shape: {scaled_features.shape}")
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        print(f"Prediction: {prediction}")
        
        # Map prediction to a risk level
        risk_levels = {
            0: "No Risk",
            1: "Very Low Risk",
            2: "Low Risk",
            3: "Medium Risk",
            4: "High Risk"
        }
        
        # Get risk level text
        risk_level = risk_levels.get(prediction, f"Unknown Risk Level ({prediction})")
        
        # Return result to user
        return render_template_string('''
        <h1>Heart Disease Risk Prediction</h1>
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-top: 20px;">
            <h2>Your Risk Level: {{ risk_level }}</h2>
            <p>Based on the information provided, our model predicts your heart disease risk level.</p>
        </div>
        <br>
        <a href="/" style="text-decoration: none; padding: 10px 15px; background-color: #4CAF50; color: white; border-radius: 4px;">Try Another Prediction</a>
        ''', risk_level=risk_level)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        print(f"Error: {str(e)}")
        print(error_details)
        
        return render_template_string('''
        <h1>Error</h1>
        <p>An error occurred: {{ error }}</p>
        <pre>{{ details }}</pre>
        <a href="/">Go back</a>
        ''', error=str(e), details=error_details)

if __name__ == '__main__':
    app.run(debug=True)