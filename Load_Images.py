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

# Map subdirectories to labels
label_map = {
    "real_photos": 0,
    "ai_generated_faces": 1,
    "human_drawn_artwork": 2
}

# Load and preprocess images
for subdirectory, label in label_map.items():
    folder_path = os.path.join(dataset_directory, subdirectory)
    
    # Check if folder exists
    if os.path.exists(folder_path):
        print(f"Processing images in: {folder_path}")
        
        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Open the image
                image = Image.open(file_path).convert("RGB")
                
                # Resize the image
                resized_image = image.resize((image_width, image_height))
                
                # Convert the image to a NumPy array
                image_array = np.array(resized_image)
                
                # Append the image data and label
                image_data.append(image_array)
                image_labels.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    else:
        print(f"Folder does not exist: {folder_path}")

# Convert the lists to NumPy arrays for easier processing later
image_data = np.array(image_data)
image_labels = np.array(image_labels)

# Print summary
print(f"Total images loaded: {len(image_data)}")
print(f"Image data shape: {image_data.shape}")
print(f"Labels shape: {image_labels.shape}")



# Create a test directory for saving files
output_dir = os.path.join(os.getcwd(), "output")
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")
except Exception as e:
    print(f"Error creating output directory: {e}")

# Save files in the output directory
image_data_path = os.path.join(output_dir, "image_data.npy")
image_labels_path = os.path.join(output_dir, "image_labels.npy")

print(f"Saving image data to: {image_data_path}")
print(f"Saving image labels to: {image_labels_path}")

try:
    np.save(image_data_path, image_data)
    print("image_data.npy saved successfully in output directory.")
    np.save(image_labels_path, image_labels)
    print("image_labels.npy saved successfully in output directory.")
except Exception as e:
    print(f"Error saving .npy files: {e}")
