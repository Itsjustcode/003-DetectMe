import numpy as np
from sklearn.model_selection import train_test_split

# Load the saved data from the output directory
image_data = np.load("output/image_data.npy")
image_labels = np.load("output/image_labels.npy")

# Split the data into training and testing sets (80% train, 20% test)
train_images, test_images, train_labels, test_labels = train_test_split(
    image_data, image_labels, test_size=0.2, random_state=42
)

# Print the sizes of the splits
print(f"Training images: {len(train_images)}, Training labels: {len(train_labels)}")
print(f"Testing images: {len(test_images)}, Testing labels: {len(test_labels)}")

# Save the split datasets for reuse
np.save("output/train_images.npy", train_images)
np.save("output/test_images.npy", test_images)
np.save("output/train_labels.npy", train_labels)
np.save("output/test_labels.npy", test_labels)

print("Train and test datasets saved successfully.")
