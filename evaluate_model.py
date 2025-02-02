import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load the test dataset
test_images = np.load("output/test_images.npy")
test_labels = np.load("output/test_labels.npy")

# Normalize test images
test_images = test_images / 255.0

# Load the trained model
model = keras.models.load_model("output/ai_face_detector.h5")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Updated class names (Only Two Classes)
class_names = ["Real Photo", "AI-Generated"]

# Generate classification report
print("\n📊 Classification Report:")
print(classification_report(test_labels, predicted_labels, target_names=class_names))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.savefig("output/evaluation_results.png")
print("📂 Evaluation results saved as 'output/evaluation_results.png'")

# Sample Predictions Plot
def plot_images(images, actual_labels, predicted_labels):
    plt.figure(figsize=(10, 5))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        predicted = class_names[predicted_labels[i]]
        actual = class_names[actual_labels[i]]
        color = "green" if predicted_labels[i] == actual_labels[i] else "red"
        plt.xlabel(f"Pred: {predicted}\nActual: {actual}", color=color)
    plt.savefig("output/sample_predictions.png")
    print("📂 Sample predictions saved as 'output/sample_predictions.png'")

# Generate predictions visualization
plot_images(test_images, test_labels, predicted_labels)
