import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted. Call fit() before transforming data.")
        standardized_data = (X - self.mean) / self.std
        return standardized_data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Example usage:
if __name__ == "__main__":
    # Sample dataset
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    # Create and fit the Standard Scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Print the original and scaled data
    print("Original Data:")
    print(data)
    print("\nScaled Data:")
    print(scaled_data)
