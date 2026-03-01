import cv2
import os
import random
import shutil

# Paths to the folders where your videos are stored
videos_folder = "C:/Users/len0v0/OneDrive/Desktop/fyp detection/surveillanceVideos"  # Adjust to your folder
activity_dataset_folder = "C:/Users/len0v0/OneDrive/Desktop/fyp detection/activity_dataset"

# Subfolders for 'train' and 'val' inside activity_dataset
train_folder = os.path.join(activity_dataset_folder, 'train')
val_folder = os.path.join(activity_dataset_folder, 'val')
temp_folder = os.path.join(activity_dataset_folder, 'temp_frames')

# Ensure base directories exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# List of events (these are your specific events, based on video name)
events = {
    "OutPW": "prowling",
    "OutLPP": "leaving_package",
    "OutPO": "passing_out",
    "OutPPP": "person_pushing",
    "OutPR": "person_running",
    "OutFG": "fighting_group",
    "OutRK": "robbery_knife",
    "OutWL": "walking"
}

# Create subdirectories for each event in 'train', 'val', and temp
for event_name in events.values():
    os.makedirs(os.path.join(train_folder, event_name), exist_ok=True)
    os.makedirs(os.path.join(val_folder, event_name), exist_ok=True)
    os.makedirs(os.path.join(temp_folder, event_name), exist_ok=True)


def extract_frames_and_split(videos_folder):
    video_files = [f for f in os.listdir(videos_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        # Infer event name from video filename using event-related patterns
        event_name = None
        for key, ev in events.items():
            if key in video_file:
                event_name = ev
                break

        if event_name is None:
            print(f"Skipping video '{video_file}' as no event is found in the filename.")
            continue

        video_path = os.path.join(videos_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        frame_files = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"{event_name}_frame_{idx:06d}.jpg"
            temp_path = os.path.join(temp_folder, event_name, frame_filename)
            cv2.imwrite(temp_path, frame)
            frame_files.append(frame_filename)
            idx += 1

        cap.release()

        if not frame_files:
            print(f"No frames extracted from {video_file}.")
            continue

        # Split frames into train and val
        random.shuffle(frame_files)
        split_index = int(0.8 * len(frame_files))
        train_files = frame_files[:split_index]
        val_files = frame_files[split_index:]

        # Move frames from temp to train/val folders
        for fname in train_files:
            src = os.path.join(temp_folder, event_name, fname)
            dst = os.path.join(train_folder, event_name, fname)
            shutil.move(src, dst)

        for fname in val_files:
            src = os.path.join(temp_folder, event_name, fname)
            dst = os.path.join(val_folder, event_name, fname)
            shutil.move(src, dst)

        print(f"Processed video '{video_file}' with {len(frame_files)} frames.")
        print(f"Moved {len(train_files)} frames to train/{event_name} and {len(val_files)} frames to val/{event_name}.")


if __name__ == '__main__':
    extract_frames_and_split(videos_folder)
