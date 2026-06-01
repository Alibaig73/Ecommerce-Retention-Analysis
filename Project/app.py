from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# 1. Load the binary serialized ML files we downloaded from Colab
model = joblib.load('best_adaboost_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Expected features in the precise layout your model was trained on
FEATURE_NAMES = [
    'Age', 'TenureMonths', 'TotalPurchases', 'AveragePurchaseValue', 
    'TotalSpend', 'MonthlySpend', 'SessionsPerMonth', 'AvgSessionDurationMinutes', 
    'PagesViewedPerSession', 'SupportTickets', 'HasPremiumMembership', 
    'LastInteractionDaysAgo', 'Gender_Male', 'DeviceUsed_Mobile', 'DeviceUsed_Tablet'
]

@app.route('/')
def home():
    # Renders your visual web landing page layout
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Extract input text values submitted via the web frontend form
        age = float(request.form['Age'])
        tenure = float(request.form['TenureMonths'])
        purchases = float(request.form['TotalPurchases'])
        avg_val = float(request.form['AveragePurchaseValue'])
        total_spend = float(request.form['TotalSpend'])
        monthly_spend = float(request.form['MonthlySpend'])
        sessions = float(request.form['SessionsPerMonth'])
        duration = float(request.form['AvgSessionDurationMinutes'])
        pages = float(request.form['PagesViewedPerSession'])
        tickets = float(request.form['SupportTickets'])
        premium = int(request.form['HasPremiumMembership'])
        last_interaction = float(request.form['LastInteractionDaysAgo'])
        
        # Parse the raw categorical dropdown fields
        gender = request.form['Gender']
        device = request.form['DeviceUsed']
        
        gender_male = 1 if gender == 'Male' else 0
        device_mobile = 1 if device == 'Mobile' else 0
        device_tablet = 1 if device == 'Tablet' else 0

        # 3. Compile variables into a clean vector array structure
        raw_features = [
            age, tenure, purchases, avg_val, total_spend, monthly_spend,
            sessions, duration, pages, tickets, premium, last_interaction,
            gender_male, device_mobile, device_tablet
        ]
        
        # Map variables into a temporary dataframe matching columns precisely
        input_df = pd.DataFrame([raw_features], columns=FEATURE_NAMES)
        
        # 4. Standard scale inputs to prevent mathematical coordinate distortion
        scaled_features = scaler.transform(input_df)
        
        # 5. Execute prediction via your custom AdaBoost brain
        prediction = model.predict(scaled_features)[0]
        
        # Translate binary integer into plain English for the web client UI view
        result = "⚠️ High Risk: Customer likely to CHURN!" if prediction == 1 else "✅ Good Standing: Customer likely to stay (RETAINED)."
        
        return render_template('index.html', prediction_text=result)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error in processing data: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)