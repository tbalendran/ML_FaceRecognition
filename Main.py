import numpy as np
import os
from skimage import io
import os
import numpy as np
from PIL import Image
from ConvLayer import ConvLayer
from ReLULayer import ReLULayer
from MaxPoolLayer import MaxPoolLayer
from FullyConnectedLayer import FullyConnectedLayer
from SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from FlattenLayer import FlattenLayer


def load_orl_faces(path="Dataset", num_subjects=40):
    X = []
    y = []

    for subject in range(1, num_subjects + 1):
        subject_path = os.path.join(path, f"s{subject}")
        for image_name in os.listdir(subject_path):
            image_path = os.path.join(subject_path, image_name)
            image = Image.open(image_path).convert("L")
            image_array = np.asarray(image, dtype=np.float32)
            X.append(image_array)
            y.append(subject - 1)

    X = np.array(X)
    y = np.array(y)

    return X, y

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def train_one_epoch(X_train, y_train, model, loss_fn, learning_rate):
        # Assuming model is a list containing the layers in order
        batch_size = X_train.shape[0]

        # Forward pass
        output = X_train
        for layer in model:
            output = layer.forward(output)

        # Calculate loss
        loss = loss_fn.forward(output, y_train)

        # Backward pass
        d_output = loss_fn.backward()
        for layer in reversed(model):
            if isinstance(layer, (ConvLayer, FullyConnectedLayer)):
                d_output, d_weights, d_biases = layer.backward(d_output)
                layer.weights -= learning_rate * d_weights
                layer.biases -= learning_rate * d_biases
            else:
                d_output = layer.backward(d_output)

        return loss

def evaluate(X_test, y_test, model):
        # Assuming model is a list containing the layers in order
        batch_size = X_test.shape[0]

        # Forward pass
        output = X_test
        for layer in model:
            output = layer.forward(output)

        # Calculate predictions
        predictions = np.argmax(output, axis=1)

        # Calculate accuracy
        accuracy = np.sum(predictions == y_test) / batch_size

        return accuracy
        
X, y = load_orl_faces(num_subjects=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rescale pixel values to 0 to 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the images to be (batch_size, 1, height, width) for the CNN
X_train = X_train.reshape(-1, 1, 112, 92)
X_test = X_test.reshape(-1, 1, 112, 92)


# Define the CNN model
model = [
    ConvLayer(1, 32, 3, 1),
    ReLULayer(),
    MaxPoolLayer(2, 2),
    ConvLayer(32, 64, 3, 1),
    ReLULayer(),
    MaxPoolLayer(2, 2),
    FlattenLayer(),
    FullyConnectedLayer(64 * 26 * 21, 128),
    ReLULayer(),
    
    FullyConnectedLayer(128, 40) # for 40 person 400 images
]


# Define the loss function
loss_fn = SoftmaxCrossEntropyLoss()

# Training parameters
epochs = 10
learning_rate = 0.01

# Train the model
for epoch in range(epochs):
    loss = train_one_epoch(X_train, y_train, model, loss_fn, learning_rate)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

# Evaluate the model
accuracy = evaluate(X_test, y_test, model)
print(f"Accuracy: {accuracy * 100:.2f}%")
