
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlsaf import MLSAF   # Import your custom activation

# Define demo model
def create_model():
    model = models.Sequential([
        layers.Conv2D(6, 5, activation=MLSAF(), input_shape=(28, 28, 1)),
        layers.AvgPool2D(2),
        layers.Conv2D(16, 5, activation=MLSAF()),
        layers.AvgPool2D(2),
        layers.Flatten(),
        layers.Dense(120, activation=MLSAF()),
        layers.Dense(84, activation=MLSAF()),
        layers.Dense(2, activation='softmax')
    ])
    return model

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = (train_labels % 2).astype('float32')
test_labels = (test_labels % 2).astype('float32')
train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Train model
model = create_model()
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
preds = np.argmax(model.predict(test_images), axis=1)

acc = accuracy_score(np.argmax(test_labels, axis=1), preds)
prec = precision_score(np.argmax(test_labels, axis=1), preds)
rec = recall_score(np.argmax(test_labels, axis=1), preds)
f1 = f1_score(np.argmax(test_labels, axis=1), preds)

tn, fp, fn, tp = confusion_matrix(np.argmax(test_labels, axis=1), preds).ravel()
specificity = tn / (tn + fp)

print(f"Accuracy: {acc:.4f}")
print(f"Recall (Sensitivity): {rec:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1 Score: {f1:.4f}")
