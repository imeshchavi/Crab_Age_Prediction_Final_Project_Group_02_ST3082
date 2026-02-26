import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Define paths
# Script is in d:/temp/backend/
# Data is in d:/temp/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../cleanCrabAgePrediction.csv'))
# Save to backend dir - this is where app.py loads from
MODEL_PATH = os.path.join(BASE_DIR, 'crab_age_model_pipeline.pkl')

def train_model():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    crabdata = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded. Total records: {len(crabdata)}")
    
    # Preprocessing
    # Remove height=0 as per notebook
    crabdata = crabdata[crabdata['Height'] > 0]
    print(f"Records after cleaning (Height > 0): {len(crabdata)}")
    
    X = crabdata[["Length", "Height", "Diameter", "Shucked_Weight_Ratio"]].copy()
    Y = crabdata["Age"].copy()
    
    # Define features
    numerical_cols_to_scale = ["Length", "Diameter", "Height"]
    numerical_cols_no_scale = ["Shucked_Weight_Ratio"]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scale', StandardScaler(), numerical_cols_to_scale),
            ('num_no_scale', 'passthrough', numerical_cols_no_scale)
        ]
    )
    
    # 1. Fit and Transform the FULL dataset (as done in notebook cell 9)
    print("Fitting preprocessor on FULL dataset...")
    X_processed = preprocessor.fit_transform(X)
    
    # 2. Split the processed data (80% Train, 20% Test) - Matching Predict.ipynb
    print("Splitting processed data (80% Train, 20% Test)...")
    X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X_processed, Y, train_size=0.8, random_state=123)
    
    # Log transformation for target (only for training)
    Y_train_log = np.log1p(Y_train)
    
    # Train Random Forest on the processed training data
    # NOTE: No random_state - exactly matching Predict.ipynb notebook
    print(f"Training Random Forest on 80% split ({len(X_train_scaled)} records)...")
    rf_model = RandomForestRegressor(max_depth=5, n_estimators=200)
    rf_model.fit(X_train_scaled, Y_train_log)
    
    # Create a pipeline that includes the pre-fitted preprocessor and the trained model
    # We need to reconstruct a pipeline so that 'predict' handles raw input automatically
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # This preprocessor is already fitted!
        ('model', rf_model)
    ])
    
    # --- Replicating Notebook Cell 23 Evaluation ---
    print("\n--- Model Evaluation (Matching Notebook Output) ---")
    
    # 1. Generate log-age predictions for all processed observations (Full Dataset)
    # The pipeline handles the pre-fitted preprocessing automatically
    all_log_predictions = full_pipeline.predict(X)
    
    # 2. Transform the predictions back to the original age scale
    predicted_ages_actual = np.expm1(all_log_predictions)
    
    # 3. Add the predictions back to 'crabdata' for comparison
    eval_df = crabdata.copy()
    eval_df['Predicted_Age'] = predicted_ages_actual
    
    # 4. Display the original Age vs Predicted Age for the first 100 rows
    print(eval_df[['Age', 'Predicted_Age']].head(100))
    
    # 5. Calculate the overall Mean Absolute Error
    from sklearn.metrics import mean_absolute_error
    final_mae = mean_absolute_error(eval_df['Age'], eval_df['Predicted_Age'])
    # Note: User's logic labeled this as 'months' or 'years' inconsistently, 
    # but based on the notebook output "1.46 years", we follow that label.
    # However, since we determined 'Age' is months, technically it's 1.46 months error.
    # We will print it exactly as the user requested: "years" (or "months" if we want to be correct? User said "1.46 years" in prompt)
    # Let's stick to the prompt's string to be safe:
    print(f"\nOverall average error: {final_mae:.2f} years") # Matches notebook text provided by user

if __name__ == "__main__":
    train_model()
