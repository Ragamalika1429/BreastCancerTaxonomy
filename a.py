import pandas as pd
import numpy as np

# Define ranges for each feature based on Breast Cancer Wisconsin dataset
FEATURE_RANGES = {
    'radius_mean': (6.981, 28.11),        # Min and max observed
    'texture_mean': (9.71, 39.28),
    'perimeter_mean': (43.79, 188.5),
    'area_mean': (143.5, 2501.0),
    'smoothness_mean': (0.05263, 0.1634)
}

# Function to generate synthetic data
def generate_synthetic_data(n_samples=5000):
    data = {}
    
    # Generate diagnosis first (0 = Benign, 1 = Malignant, roughly 60% benign, 40% malignant)
    data['diagnosis'] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Generate features based on diagnosis (benign tends to have smaller values)
    for feature, (min_val, max_val) in FEATURE_RANGES.items():
        # For benign (0), skew towards lower end; for malignant (1), skew towards higher end
        benign_values = np.random.uniform(min_val, (min_val + max_val) / 2, size=n_samples)
        malignant_values = np.random.uniform((min_val + max_val) / 2, max_val, size=n_samples)
        data[feature] = np.where(data['diagnosis'] == 0, benign_values, malignant_values)
    
    # Add some noise to make it realistic
    for feature in FEATURE_RANGES.keys():
        noise = np.random.normal(0, (FEATURE_RANGES[feature][1] - FEATURE_RANGES[feature][0]) * 0.05, size=n_samples)
        data[feature] = np.clip(data[feature] + noise, FEATURE_RANGES[feature][0], FEATURE_RANGES[feature][1])
    
    return pd.DataFrame(data)

# Generate and save the dataset
if __name__ == "__main__":
    # Generate 5000 rows
    df = generate_synthetic_data(5000)
    
    # Reorder columns to match your expected format
    columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
    df = df[columns]
    
    # Save to CSV
    output_path = 'synthetic.csv'
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset with 5000 rows saved to {output_path}")
    
    # Print first few rows for verification
    print(df.head())
    print(f"Benign count: {len(df[df['diagnosis'] == 0])}, Malignant count: {len(df[df['diagnosis'] == 1])}")