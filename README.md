# Car Price Prediction ML Pipeline

## Overview

This project implements an end-to-end machine learning pipeline to predict car selling prices using a dataset obtained from Kaggle. The pipeline encompasses data loading, preprocessing (including exploratory data analysis, handling missing values, categorical encoding, and feature scaling), model training with hyperparameter tuning using GridSearchCV, evaluation using standard regression metrics, and deployment via a Flask API. The end goal is to create a robust, explainable model accessible through a simple web interface.

## Repository Structure

```
car-price-ml/
├── data/                 
│   └── car_prediction_data.csv     # Local copy of the dataset (optional)
├── models/               
│   ├── car_price_model.pkl           # Trained ML model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   └── training_columns.pkl          # List of training features after one-hot encoding
├── src/                  
│   └── model_training.py             # Python script for data loading, preprocessing, model training, and evaluation
├── notebooks/            
│   └── model_training.ipynb          # Jupyter Notebook version of the model training script
├── api/                  
│   └── app.py                        # Flask API code for serving predictions
├── README.md                         # This file
└── requirements.txt                  # Project dependencies
```

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/dheerajkreddy/car-price-prediction-ml.git
   cd car-price-prediction-ml
   ```

2. **(Optional) Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Model Training

The main model training script is located in the `src/` directory. This script:

- Downloads the dataset using the Kaggle API.
- Performs exploratory data analysis (EDA) to understand the dataset.
- Preprocesses the data (one-hot encoding for categorical variables, scaling, etc.).
- Trains a Random Forest Regressor with hyperparameter tuning via GridSearchCV.
- Evaluates the model using metrics such as MAE, MSE, RMSE, and R².
- Saves the model and preprocessing artifacts in the `models/` folder.

Run the training script:

```bash
python src/model_training.py
```

### API Deployment

The Flask API is contained in the `api/app.py` file. It:

- Loads the saved model, scaler, and training columns.
- Provides a simple web interface with an HTML form to enter raw input features.
- Processes the input (one-hot encoding, reindexing, scaling) and returns a prediction.

To run the API:

```bash
python api/app.py
```

Then, access the API at: [http://localhost:5001](http://localhost:5001)

## How to Use

1. **Data Input:**  
   Fill out the form on the home page with the following fields:
   - **Car_Name:** e.g., "ritz"
   - **Year:** e.g., 2014
   - **Present_Price (in Lakhs Rupees):** e.g., 5.59
   - **Kms_Driven:** e.g., 27000
   - **Fuel_Type:** e.g., "Petrol"
   - **Seller_Type:** e.g., "Dealer"
   - **Transmission:** e.g., "Manual"
   - **Owners:** e.g., 0

2. **Prediction:**  
   Click the “Predict Selling Price” button. The API processes the input and returns the predicted selling price in an HTML response.

## References

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Kuhn, M., & Johnson, K. (2013). *Applied predictive modeling*. Springer.
- Bhavikjikadara. (n.d.). *Car Price Prediction Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/bhavikjikadara/car-price-prediction-dataset

## License

This project is provided for educational purposes.

## Contact

For any questions or suggestions, please contact dheeraj.aug12@gmail.com.