import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def load_data(filepath):
    data = pd.read_csv(filepath)
    # Replace 'error' and empty strings with NaN
    data.replace(['error', ''], np.nan, inplace=True)
    # Convert all columns except 'Cultivo' to numeric
    for col in data.columns:
        if col != 'Cultivo':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    # Drop rows with any missing values
    data = data.dropna()
    return data

def preprocess_data(data):
    label_encoder = LabelEncoder()
    data['Cultivo'] = label_encoder.fit_transform(data['Cultivo'])
    return data

def main():
    # Load and clean dataset
    data = load_data('dataset_agricultura.csv')
    
    # Encode target
    data = preprocess_data(data)
    
    # Split data into features and target
    X = data.drop('Cultivo', axis=1)
    y = data['Cultivo']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()