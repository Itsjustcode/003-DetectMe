import numpy as np
from sklearn.model_selection import train_test_split

# Load the saved data
image_data = np.load("output/image_data.npy")
image_labels = np.load("output/image_labels.npy")

# Ensure balanced splitting
train_images, test_images, train_labels, test_labels = train_test_split(
    image_data, image_labels, test_size=0.2, random_state=42, stratify=image_labels
)

# Convert lists to NumPy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Save the split datasets
np.save("output/train_images.npy", train_images)
np.save("output/test_images.npy", test_images)
np.save("output/train_labels.npy", train_labels)
np.save("output/test_labels.npy", test_labels)

print(f"✅ Data split complete: Train={len(train_images)}, Test={len(test_images)}")
