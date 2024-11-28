import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load or create the dataset file
dataset_path = 'refined_study_hours_scores_v3.csv'

if os.path.exists(dataset_path):
    data = pd.read_csv(dataset_path)
else:
    # Create an initial empty dataset if it doesn't exist
    data = pd.DataFrame(columns=["StudyHours", "Score"])

# Define a function to retrain the model with updated data
def retrain_model(data):
    X = data[['Score']]
    y = data['StudyHours']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model Mean Squared Error: {mse:.2f}")
    print(f"Model R² Score (Confidence): {r2:.2f}")
    return model, r2

# Initial training
model, r2 = retrain_model(data)
prediction_count = 0

# Interactive prediction loop with periodic retraining
while True:
    score = input("Enter score to predict study hours (or type 'quit' to exit): ")
    if score.lower() == 'quit':
        break
    score = float(score)
    score_df = pd.DataFrame([[score]], columns=["Score"])
    predicted_hours = model.predict(score_df)
    print(f"Predicted Study Hours: {predicted_hours[0]:.2f}")
    print(f"Model Confidence (R² Score): {r2:.2f}")
    
    # Ask for user feedback to collect correct data if the prediction is incorrect
    feedback = input("Was this prediction correct? (y/n): ").strip().lower()
    if feedback == 'n':
        actual_hours = float(input("Enter the correct number of study hours: "))
        
        # Append the corrected data to the dataset
        new_data = pd.DataFrame({"StudyHours": [actual_hours], "Score": [score]})
        data = pd.concat([data, new_data], ignore_index=True)
        
        # Sort data by Score in ascending order and save it
        data = data.sort_values(by="Score").reset_index(drop=True)
        data.to_csv(dataset_path, index=False)
        print("Dataset updated with corrected data, sorted by Score.")

    # Increment the prediction counter and retrain the model every 5 predictions
    prediction_count += 1
    if prediction_count % 5 == 0:
        print("Retraining model with updated data...")
        model, r2 = retrain_model(data)
        print("Model retrained with updated data.")

    print()  # Add space between predictions for readability
