import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path, sep='\s+', header=None)
    df.columns = [f'Feature_{i}' for i in range(1, 25)] + ['Target']

    # Strip whitespace from column names (just in case)
    df.columns = df.columns.str.strip()

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Remove ID-like columns (e.g., all unique or named "ID")
    for col in df.columns:
        if 'id' in col.lower() or df[col].nunique() == len(df):
            print(f"Dropping ID-like column: {col}")
            df.drop(col, axis=1, inplace=True)

    # Split features and target
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print("\nAfter SMOTE:", Counter(y_resampled))

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data('german.data-numeric')
