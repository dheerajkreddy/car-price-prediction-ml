from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved artifacts
model = joblib.load('./models/car_price_model.pkl')
scaler = joblib.load('./models/scaler.pkl')
training_columns = joblib.load('./models/training_columns.pkl')

# HTML for the home page with a form
home_page = """
<!DOCTYPE html>
<html>
  <head>
    <title>Car Price Prediction</title>
  </head>
  <body>
    <h1>Welcome to the Car Price Prediction App</h1>
    <form action="/predict" method="post">
      <label for="Car_Name">Car Name:</label>
      <input type="text" id="Car_Name" name="Car_Name" required><br><br>
      
      <label for="Year">Year:</label>
      <input type="number" id="Year" name="Year" required><br><br>
      
      <label for="Present_Price">Present Price(Lakhs Rupees):</label>
      <input type="text" id="Present_Price" name="Present_Price" required><br><br>
      
      <label for="Kms_Driven">Kms Driven:</label>
      <input type="number" id="Kms_Driven" name="Kms_Driven" required><br><br>
      
      <label for="Fuel_Type">Fuel Type:</label>
      <input type="text" id="Fuel_Type" name="Fuel_Type" required><br><br>
      
      <label for="Seller_Type">Seller Type:</label>
      <input type="text" id="Seller_Type" name="Seller_Type" required><br><br>
      
      <label for="Transmission">Transmission:</label>
      <input type="text" id="Transmission" name="Transmission" required><br><br>
      
      <label for="Owner">Owner:</label>
      <input type="number" id="Owner" name="Owner" required><br><br>
      
      <input type="submit" value="Predict Selling Price">
    </form>
  </body>
</html>
"""

@app.route('/')
def home():
    return home_page

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get raw input from the form
        data = request.form.to_dict()
        
        # Convert numeric fields to float (adjust if necessary)
        data['Year'] = float(data['Year'])
        data['Present_Price'] = float(data['Present_Price'])
        data['Kms_Driven'] = float(data['Kms_Driven'])
        data['Owner'] = float(data['Owner'])
        
        # Convert raw input to a DataFrame
        input_df = pd.DataFrame([data])
        print("Raw input received:")
        print(input_df)
        
        # Manually perform one-hot encoding on the raw input
        input_processed = pd.get_dummies(input_df)
        # Reindex to match the training set features, filling missing columns with 0
        input_processed = input_processed.reindex(columns=training_columns, fill_value=0)
        
        # Scale the processed input using the saved scaler
        input_scaled = scaler.transform(input_processed)
        
        # Predict using the loaded model
        prediction = model.predict(input_scaled)[0]
        
        # Return the prediction within a simple HTML response
        result_page = f"""
        <!DOCTYPE html>
        <html>
          <head>
            <title>Prediction Result</title>
          </head>
          <body>
            <h1>Predicted Selling Price (Lakhs Rupees): {prediction:.2f}</h1>
            <a href="/">Go back</a>
          </body>
        </html>
        """
        return result_page
    else:
        return home_page

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
