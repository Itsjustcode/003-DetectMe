import os
from PIL import Image
import numpy as np

# Define the base dataset directory
dataset_directory = "dataset"

# Standard image size for processing
image_width = 128
image_height = 128

# Dictionary to store processed images and their labels
image_data = []
image_labels = []

# **Updated label map (Removed Human-Drawn Artwork)**
label_map = {
    "real_photos": 0,
    "ai_generated_faces": 1
}

# Load and preprocess images
for subdirectory, label in label_map.items():
    folder_path = os.path.join(dataset_directory, subdirectory)
    
    if os.path.exists(folder_path):
        print(f"Processing images in: {folder_path}")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                image = Image.open(file_path).convert("RGB")
                resized_image = image.resize((image_width, image_height))
                image_array = np.array(resized_image)
                image_data.append(image_array)
                image_labels.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Convert lists to NumPy arrays
image_data = np.array(image_data)
image_labels = np.array(image_labels)

# Print summary
print(f"Total images loaded: {len(image_data)}")
print(f"Image data shape: {image_data.shape}")
print(f"Labels shape: {image_labels.shape}")

# Create the output directory if it doesn’t exist
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# Save processed data
np.save("output/image_data.npy", image_data)
np.save("output/image_labels.npy", image_labels)

print("✅ Image data and labels saved successfully.")
