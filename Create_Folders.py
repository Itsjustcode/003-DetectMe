import os

# Define the base directory for the dataset
dataset_directory = "dataset"

# Define subdirectories for different image types
subdirectories = ["real_photos", "ai_generated_faces", "human_drawn_artwork"]

# Create the directories if they don't already exist
for subdirectory in subdirectories:
    path = os.path.join(dataset_directory, subdirectory)
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")
