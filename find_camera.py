import os
import random
import shutil

# Paths to the folders where your normal and augmented frames are stored
normal_frames_folder = "C:/Users/len0v0/OneDrive/Desktop/fyp detection/extracted_frames"  # Adjust to your folder
augmented_frames_folder = "C:/Users/len0v0/OneDrive/Desktop/fyp detection/augmented_frames"  # Adjust to your folder

# Path to the activity_dataset folder, which already contains 'train' and 'val'
activity_dataset_folder = "C:/Users/len0v0/OneDrive/Desktop/fyp detection/activity_dataset"

# Subfolders for 'train' and 'val' inside activity_dataset
train_folder = os.path.join(activity_dataset_folder, 'train')
val_folder = os.path.join(activity_dataset_folder, 'val')

# Ensure 'train' and 'val' directories exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Function to create event subfolders for 'train' and 'val'
def create_event_subfolders(base_folder):
    event_categories = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    print(f"Creating event subfolders in {base_folder}: {event_categories}")
    for event in event_categories:
        os.makedirs(os.path.join(train_folder, event), exist_ok=True)
        os.makedirs(os.path.join(val_folder, event), exist_ok=True)

# Create subdirectories for each event type in 'train' and 'val'
create_event_subfolders(normal_frames_folder)
create_event_subfolders(augmented_frames_folder)

# Function to split and move frames into 'train' and 'val' based on the event
def split_and_move_frames(event_folder, base_output_folder):
    event_name = event_folder.split(os.sep)[-1]  # Extract event name (e.g., "Walking")
    
    # Get all frame files for the event (e.g., "Walking", "Running")
    frame_files = [f for f in os.listdir(event_folder) if f.endswith(".jpg")]
    print(f"Found {len(frame_files)} frames for event '{event_name}'.")

    # Check if the event folder contains any frames
    if not frame_files:
        print(f"Warning: No frames found in the event folder '{event_name}'. Skipping.")
        return

    # Shuffle the frames for random splitting
    random.shuffle(frame_files)

    # Split into 80% train and 20% validation
    split_index = int(0.8 * len(frame_files))
    train_files = frame_files[:split_index]
    val_files = frame_files[split_index:]

    print(f"Moving {len(train_files)} frames to the train folder and {len(val_files)} frames to the val folder.")

    # Move the training files to the 'train' folder
    for frame_file in train_files:
        src = os.path.join(event_folder, frame_file)
        dst = os.path.join(base_output_folder, 'train', event_name, frame_file)
        shutil.move(src, dst)

    # Move the validation files to the 'val' folder
    for frame_file in val_files:
        src = os.path.join(event_folder, frame_file)
        dst = os.path.join(base_output_folder, 'val', event_name, frame_file)
        shutil.move(src, dst)

    print(f"Dataset split for event '{event_name}': {len(train_files)} training frames, {len(val_files)} validation frames.")

# Process both normal frames and augmented frames
def process_frames():
    # Process normal frames
    print("Processing normal frames...")
    for event_folder in os.listdir(normal_frames_folder):
        event_path_normal = os.path.join(normal_frames_folder, event_folder)
        if os.path.isdir(event_path_normal):
            print(f"Processing normal frames for event: {event_folder}")
            split_and_move_frames(event_path_normal, base_output_folder=normal_frames_folder)

    # Process augmented frames
    print("Processing augmented frames...")
    for event_folder in os.listdir(augmented_frames_folder):
        event_path_augmented = os.path.join(augmented_frames_folder, event_folder)
        if os.path.isdir(event_path_augmented):
            print(f"Processing augmented frames for event: {event_folder}")
            split_and_move_frames(event_path_augmented, base_output_folder=augmented_frames_folder)

    print("Splitting and moving frames complete.")

# Run the process
process_frames()
