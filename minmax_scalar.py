import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)

    def transform(self, X):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler not fitted. Call fit() before transforming data.")
        
        scaled_data = (X - self.min_val) / (self.max_val - self.min_val)
        scaled_data = scaled_data * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled_data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Example usage:
if __name__ == "__main__":
    # Sample dataset
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    # Create and fit the Min-Max Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Print the original and scaled data
    print("Original Data:")
    print(data)
    print("\nScaled Data:")
    print(scaled_data)

