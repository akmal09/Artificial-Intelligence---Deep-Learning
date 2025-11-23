import numpy as np

# --- Helper functions ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is already sigmoid(x), so derivative = x * (1 - x)
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    # add small epsilon to avoid log(0)
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# --- Simple Neural Network class ---
# data -> input layer -> hidden layer -> output layer
class SimpleANN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        self.lr = learning_rate

        # Xavier-ish initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # store intermediate values for backprop
        self.z1 = X @ self.W1 + self.b1          # (N, hidden_dim)
        self.a1 = sigmoid(self.z1)              # hidden activation

        self.z2 = self.a1 @ self.W2 + self.b2   # (N, output_dim)
        self.a2 = sigmoid(self.z2)              # output activation

        return self.a2

    def backward(self, X, y):
        """
        Backprop using:
        - output layer: sigmoid + BCE -> dL/dz2 = (a2 - y)
        """

        m = X.shape[0]          # number of samples
        y_pred = self.a2        # (N, 1)

        # --- Gradients for output layer ---
        dz2 = (y_pred - y)      # (N, 1)
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # --- Gradients for hidden layer ---
        da1 = dz2 @ self.W2.T                    # (N, hidden_dim)
        dz1 = da1 * sigmoid_derivative(self.a1)  # (N, hidden_dim)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # --- Gradient descent update ---
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=10000, print_every=1000):
        for epoch in range(1, epochs + 1):
            # forward
            y_pred = self.forward(X)

            # compute loss
            loss = binary_cross_entropy(y, y_pred)

            # backward
            self.backward(X, y)

            if epoch % print_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int), y_pred


# --- Example usage: XOR problem ---

if __name__ == "__main__":
    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    np.random.seed(42)

    model = SimpleANN(input_dim=2, hidden_dim=2, output_dim=1, learning_rate=0.1)
    model.train(X, y, epochs=10000, print_every=1000)

    preds_label, preds_prob = model.predict(X)
    print("\nFinal predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}  ->  predicted: {preds_label[i][0]}  (prob={preds_prob[i][0]:.4f})  | target: {y[i][0]}")
