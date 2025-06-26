import numpy as np
import pandas as pd
import pickle
import shap
import lime
import lime.lime_tabular
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Load trained RandomForest model
model = pickle.load(open("RFC_Model.pkl", "rb"))

# Define features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
                        'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
                        'StreamingMovies', 'StreamingTV']
feature_names = numerical_features + categorical_features

# Load and preprocess training data
X_train_raw = pd.read_csv("Telco-Customer-Churn.csv")
X_train_raw['TotalCharges'] = pd.to_numeric(X_train_raw['TotalCharges'], errors='coerce')

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    le.fit(X_train_raw[col].astype(str))
    label_encoders[col] = le

X_train_processed = X_train_raw[feature_names].copy()
X_train_processed[numerical_features] = X_train_processed[numerical_features].apply(pd.to_numeric, errors='coerce')
X_train_processed.dropna(inplace=True)

for col in categorical_features:
    X_train_processed[col] = label_encoders[col].transform(X_train_processed[col].astype(str))

# Rule-based logic
def rule_based_risk(form_data):
    try:
        high_risk_conditions = [
            form_data['Contract'] == 'Month-to-month',
            form_data['TechSupport'] == 'No',
            form_data['OnlineSecurity'] == 'No',
            form_data['InternetService'] == 'Fiber optic',
            form_data['PaymentMethod'] == 'Electronic check',
            form_data['DeviceProtection'] == 'No',
            form_data['OnlineBackup'] == 'No',
            form_data['StreamingMovies'] == 'Yes',
            form_data['StreamingTV'] == 'Yes',
            float(form_data['tenure']) < 6,
            float(form_data['MonthlyCharges']) > 80,
            float(form_data['TotalCharges']) < 200,
        ]

        medium_risk_conditions = [
            form_data['Contract'] == 'One year',
            form_data['TechSupport'] == 'No',
            form_data['OnlineSecurity'] == 'No',
            form_data['InternetService'] == 'DSL',
            form_data['DeviceProtection'] == 'No',
            6 <= float(form_data['tenure']) < 12,
            60 <= float(form_data['MonthlyCharges']) <= 80,
            200 <= float(form_data['TotalCharges']) < 500,
        ]

        high_score = sum(high_risk_conditions)
        medium_score = sum(medium_risk_conditions)

        if high_score >= 6:
            return "High"
        elif medium_score >= 4:
            return "Medium"
        else:
            return "Low"
    except Exception as e:
        print(f"Rule-based risk calculation error: {e}")
        return "Unknown"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------- 1. Input preprocessing ----------
        input_data = []
        for feature in feature_names:
            value = request.form[feature]
            if feature in numerical_features:
                input_data.append(float(value))
            else:
                encoder = label_encoders[feature]
                if value in encoder.classes_:
                    input_data.append(encoder.transform([value])[0])
                else:
                    return f"Invalid value '{value}' for feature '{feature}'"

        input_df = pd.DataFrame([input_data], columns=feature_names)

        # ---------- 2. Model prediction ----------
        prediction_proba = model.predict_proba(input_df)
        risk_score = prediction_proba[0][1]  # Probability of churn
        risk_category = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"

        if risk_score > 0.7:
            retention_action = ['Grant loyalty benefits', 'Offer cashback offers', 'Schedule agent call to customer']
        elif risk_score > 0.3:
            retention_action = ['Grant loyalty points']
        else:
            retention_action = ['No Action Required']

        # ---------- 3. Rule-based category ----------
        rule_based_category = rule_based_risk(request.form)

        # ---------- 4. LIME Explanation ----------
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train_processed),
                feature_names=feature_names,
                class_names=['No Churn', 'Churn'],
                mode='classification'
            )

            lime_explanation = lime_explainer.explain_instance(
                input_df.values[0],
                model.predict_proba,
                num_features=4
            )

            lime_html = lime_explanation.as_html()
        except Exception as e:
            print(f"LIME explanation error: {e}")
            lime_html = f"<p>LIME explanation unavailable: {str(e)}</p>"

        # ---------- 5. SHAP Analysis (FIXED) ----------
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            print(f"Debug - SHAP values type: {type(shap_values)}")
            print(f"Debug - SHAP values structure: {shap_values}")
            print(f"Debug - Expected value: {explainer.expected_value}")

            # Handle different SHAP output formats
            if isinstance(shap_values, list):  # Binary classification returns list
                print(f"Debug - List format, length: {len(shap_values)}")
                print(f"Debug - Shape of shap_values[1]: {shap_values[1].shape}")
                
                # For binary classification, use the positive class (index 1)
                shap_val = shap_values[1][0]  # First sample, positive class
                
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    base_val = explainer.expected_value[1]
                else:
                    base_val = explainer.expected_value
            else:
                print(f"Debug - Array format, shape: {shap_values.shape}")
                # Single array output
                shap_val = shap_values[0]
                base_val = explainer.expected_value

            print(f"Debug - shap_val shape: {shap_val.shape}")
            print(f"Debug - base_val: {base_val}")

            # Ensure shap_val is 1D array
            if len(shap_val.shape) > 1:
                shap_val = shap_val.flatten()

            # Convert arrays to proper format for processing
            shap_val = np.array(shap_val).flatten()
            
            # Ensure static/ directory exists
            Path("static").mkdir(exist_ok=True)

            # Create SHAP feature impact table
            feature_impacts = []
            for i, feature in enumerate(feature_names):
                if i < len(shap_val):
                    feature_impacts.append((feature, shap_val[i]))
                else:
                    print(f"Warning: Missing SHAP value for feature {feature}")
                    feature_impacts.append((feature, 0.0))
            
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Convert numpy values to Python scalars for string formatting
            # Handle potential scalar vs array issues
            try:
                if np.isscalar(base_val):
                    base_val_scalar = float(base_val)
                else:
                    base_val_scalar = float(base_val.item() if hasattr(base_val, 'item') else base_val[0])
                
                total_impact = float(np.sum(shap_val))
                final_prediction = base_val_scalar + total_impact
            except Exception as scalar_error:
                print(f"Scalar conversion error: {scalar_error}")
                base_val_scalar = 0.5  # fallback value
                total_impact = 0.0
                final_prediction = base_val_scalar
            
            shap_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .shap-container {{ max-width: 800px; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                    .positive {{ color: #dc3545; font-weight: bold; }}
                    .negative {{ color: #007bff; font-weight: bold; }}
                    .bar {{ height: 20px; border-radius: 3px; margin: 2px 0; }}
                    .bar-positive {{ background-color: #ffebee; }}
                    .bar-negative {{ background-color: #e3f2fd; }}
                    .bar-fill {{ height: 100%; border-radius: 3px; }}
                    .bar-fill-positive {{ background-color: #dc3545; }}
                    .bar-fill-negative {{ background-color: #007bff; }}
                </style>
            </head>
            <body>
                <div class="shap-container">
                    <div class="summary">
                        <h3>SHAP Analysis Summary</h3>
                        <p><strong>Base Value (average prediction):</strong> {base_val_scalar:.4f}</p>
                        <p><strong>Final Prediction:</strong> {final_prediction:.4f}</p>
                        <p><strong>Total Impact:</strong> {total_impact:+.4f}</p>
                        <p><strong>Risk Score:</strong> {risk_score:.4f} ({risk_category})</p>
                    </div>
                    
                    <h4>Feature Contributions (ordered by impact)</h4>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Value</th>
                            <th>SHAP Impact</th>
                            <th>Visual Impact</th>
                        </tr>
            """
            
            max_abs_impact = 1.0
            if feature_impacts:
                impacts = [abs(float(impact)) for _, impact in feature_impacts]
                if impacts:
                    max_abs_impact = max(impacts)
            
            for i, (feature, impact) in enumerate(feature_impacts):
                try:
                    # Convert to Python scalars for string formatting
                    feature_value = input_df.iloc[0][feature]
                    
                    # Handle different types of feature values
                    if hasattr(feature_value, 'item'):
                        feature_value = feature_value.item()
                    feature_value = float(feature_value)
                    
                    # Handle impact conversion
                    if hasattr(impact, 'item'):
                        impact_scalar = impact.item()
                    else:
                        impact_scalar = float(impact)
                    
                    impact_class = "positive" if impact_scalar > 0 else "negative"
                    bar_class = "bar-positive" if impact_scalar > 0 else "bar-negative"
                    bar_fill_class = "bar-fill-positive" if impact_scalar > 0 else "bar-fill-negative"
                    bar_width = (abs(impact_scalar) / max_abs_impact) * 100 if max_abs_impact > 0 else 0
                    
                    shap_html += f"""
                        <tr>
                            <td><strong>{feature}</strong></td>
                            <td>{feature_value:.4f}</td>
                            <td class="{impact_class}">{impact_scalar:+.4f}</td>
                            <td>
                                <div class="bar {bar_class}">
                                    <div class="bar-fill {bar_fill_class}" style="width: {bar_width:.1f}%;"></div>
                                </div>
                            </td>
                        </tr>
                    """
                except Exception as row_error:
                    print(f"Error processing feature {feature}: {row_error}")
                    # Add a fallback row
                    shap_html += f"""
                        <tr>
                            <td><strong>{feature}</strong></td>
                            <td>Error</td>
                            <td>Error</td>
                            <td>Error processing</td>
                        </tr>
                    """
                    continue
            
            shap_html += """
                    </table>
                    <div style="margin-top: 20px; font-size: 14px; color: #666;">
                        <p><strong>How to read this:</strong></p>
                        <ul>
                            <li><span style="color: #dc3545;">Red values</span> push the prediction toward churn (increase risk)</li>
                            <li><span style="color: #007bff;">Blue values</span> push the prediction away from churn (decrease risk)</li>
                            <li>Longer bars indicate stronger influence on the prediction</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """

            # Save SHAP HTML
            with open("static/shap_force_plot.html", "w") as f:
                f.write(shap_html)

        except Exception as e:
            print(f"SHAP analysis error: {e}")
            # Create fallback SHAP HTML
            shap_html = f"""
            <html>
            <body>
                <div style="padding: 20px;">
                    <h3>SHAP Analysis Unavailable</h3>
                    <p>Error: {str(e)}</p>
                    <p>Risk Score: {risk_score:.4f} ({risk_category})</p>
                </div>
            </body>
            </html>
            """
            with open("static/shap_force_plot.html", "w") as f:
                f.write(shap_html)

        # ---------- 6. Return HTML ----------
        return render_template("predict.html",
                               risk_score=round(risk_score * 100, 2),
                               risk_category=risk_category,
                               retention_actions=retention_action,
                               rule_based_category=rule_based_category,
                               lime_html=lime_html)

    except Exception as e:
        print(f"General prediction error: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)