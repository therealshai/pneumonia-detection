#%%
import os
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Import here as it's used in main
#%%
# --- Configuration ---

PROJECT_ROOT = r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local"
EXTERNAL_DATASET_DIR = os.path.join(PROJECT_ROOT, "external-dataset")
METADATA_FILE = os.path.join(EXTERNAL_DATASET_DIR, "meta_data.csv")

# Directories where original images are stored
# If your images are directly under EXTERNAL_DATASET_DIR, update accordingly. Adjust if subfolders exist.
IMAGE_SOURCE_DIRS = [os.path.join(EXTERNAL_DATASET_DIR, f"images_{i:03d}", "images") for i in range(1, 13)]

# Target directories for the organized dataset
TRAIN_DIR = os.path.join(EXTERNAL_DATASET_DIR, "train")
TEST_DIR = os.path.join(EXTERNAL_DATASET_DIR, "test")
VAL_DIR = os.path.join(EXTERNAL_DATASET_DIR, "val")

# --- Helper Functions ---
#%%
def find_image_in_subfolders(image_name, source_dirs):
    """
    Finds the full path of an image within a list of potential source directories.
    """
    for s_dir in source_dirs:
        image_path = os.path.join(s_dir, image_name)
        if os.path.exists(image_path):
            return image_path
    return None
#%%
def organize_dataset_physically(metadata_df, target_base_dir, source_dirs, split_name):
    """
    Organizes images into 'normal' and 'pneumonia' subfolders within a given split directory.
    metadata_df: DataFrame containing Image Index and Finding Labels for a specific split (train/val/test).
    target_base_dir: The root directory for the split (e.g., TRAIN_DIR, VAL_DIR, TEST_DIR).
    source_dirs: List of directories where original images are located (e.g., images_001/images to images_012/images).
    split_name: Name of the split (e.g., 'train', 'val', 'test') for logging.
    """
    print(f"\n--- Organizing {split_name} split ---")
    normal_dir = os.path.join(target_base_dir, "normal")
    pneumonia_dir = os.path.join(target_base_dir, "pneumonia")

    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)

    moved_count = 0
    skipped_count = 0

    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Moving {split_name} images"):
        image_name = row['Image Index']
        finding_label = row['Finding Labels']

        # Determine destination folder
        if 'Pneumonia' in finding_label: # This should now only be 'Pneumonia' or 'Normal'
            dest_dir = pneumonia_dir
        else: # Assumes 'Normal'
            dest_dir = normal_dir

        original_image_path = find_image_in_subfolders(image_name, source_dirs)

        if original_image_path:
            destination_path = os.path.join(dest_dir, image_name)
            if not os.path.exists(destination_path): # Prevent re-moving if script is re-run
                try:
                    # Use shutil.copy2 instead of os.rename to keep original files in source folders
                    # and allow data_loaders.py to access them during its DataFrame split.
                    # If you truly want to *move* (delete from source), use shutil.move.
                    # For a robust pipeline, copying is safer so the source is preserved.
                    shutil.copy2(original_image_path, destination_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error copying {image_name} to {destination_path}: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1 # Count as skipped if already there
        else:
            print(f"Warning: Image {image_name} not found in any source directory. Skipping.")
            skipped_count += 1
            
    print(f"Finished organizing {split_name}. Moved {moved_count} images, Skipped {skipped_count} images.")
#%%
# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading metadata from: {METADATA_FILE}")
    metadata = pd.read_csv(METADATA_FILE)

    # --- FIX: More robust filtering for 'Pneumonia' and 'Normal' classes ---
    # First, standardize 'No Finding' to 'Normal'
    metadata['Finding Labels'] = metadata['Finding Labels'].replace('No Finding', 'Normal')

    # Now, explicitly filter to include only 'Pneumonia' and 'Normal'
    initial_filtered_metadata = metadata[
        (metadata['Finding Labels'] == 'Pneumonia') |
        (metadata['Finding Labels'] == 'Normal')
    ].copy()
    # --- END FIX ---

    print(f"Total relevant images in metadata after initial filtering: {len(initial_filtered_metadata)}")
    print("\nClass distribution before splitting:")
    print(initial_filtered_metadata['Finding Labels'].value_counts())

    # Perform the train/val/test split using sklearn's train_test_split
    # Note: Ensure you have enough samples for stratification in smaller splits.
    
    # First split into train and temp (for val/test)
    train_df, temp_df = train_test_split(
        initial_filtered_metadata,
        test_size=0.3, # 30% for temp (val+test)
        stratify=initial_filtered_metadata['Finding Labels'],
        random_state=42
    )

    # Then split temp into val and test (50/50 of temp, so 15% each of original)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5, # 50% of temp_df goes to test, 50% to val
        stratify=temp_df['Finding Labels'],
        random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")

    # Now, physically move the images based on these DataFrames
    print("\nStarting physical organization of files...")
    organize_dataset_physically(train_df, TRAIN_DIR, IMAGE_SOURCE_DIRS, 'train')
    organize_dataset_physically(val_df, VAL_DIR, IMAGE_SOURCE_DIRS, 'val')
    organize_dataset_physically(test_df, TEST_DIR, IMAGE_SOURCE_DIRS, 'test')

    print("\nDataset organization complete!")


# %%
