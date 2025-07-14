#%%
import os
import pandas as pd

# Root dir where images are already split
base_dir = r"C:/Users/SyedShaistaTabassum/Capstone project/pneumonia-detection-local/external-dataset"

# %%
met_data = pd.read_csv(os.path.join(base_dir, "meta_data.csv"))
# %%
met_data.head()
# %%
# Filter out rows from finding labels = Pneumonia
met_data_normal = met_data[met_data['Finding Labels'].str.contains('Pneumonia') == False]
met_data_normal.sum(axis=0)

# %%
met_data_Pneumonia = met_data[met_data['Finding Labels'] == 'Pneumonia']
met_data_Pneumonia.sum(axis=0)
met_data_Pneumonia.shape
# %% 
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "val")
# %% 
# MAP the finding labels to the directory names
label_map = {
    "Pneumonia": "pneumonia",
    "Normal": "normal"
}
# %%
# Create a subfolder in each directory for each label
for dir_path in [train_dir, test_dir, val_dir]:
    for label in label_map.values():
        os.makedirs(os.path.join(dir_path, label), exist_ok=True)
# %%
# Move images to their respective directories based on labels

def move_images_for_label(df, label, split_dir, base_dir):
    """
    Moves images from split_dir to split_dir/label for all images in df.
    """
    for index, row in df.iterrows():
        image_path = os.path.join(split_dir, row['Image Index'])
        if os.path.exists(image_path):
            dest_dir = os.path.join(split_dir, label)
            os.makedirs(dest_dir, exist_ok=True)
            os.rename(image_path, os.path.join(dest_dir, row['Image Index']))
# %%
# Move normal images for each split
move_images_for_label(met_data_normal, "normal", train_dir, base_dir)
move_images_for_label(met_data_normal, "normal", test_dir, base_dir)
move_images_for_label(met_data_normal, "normal", val_dir, base_dir)
# %%
# Move pneumonia images for each split
move_images_for_label(met_data_Pneumonia, "pneumonia", train_dir, base_dir)
move_images_for_label(met_data_Pneumonia, "pneumonia", test_dir, base_dir)
move_images_for_label(met_data_Pneumonia, "pneumonia", val_dir, base_dir)

# %%
# Check the number of normal samples in metadata
print("Number of normal samples in metadata:", len(met_data_normal))

# Count number of images in 'normal' subfolders of train and test directories
train_normal_count = len(os.listdir(os.path.join(train_dir, "normal")))
test_normal_count = len(os.listdir(os.path.join(test_dir, "normal")))
val_normal_count = len(os.listdir(os.path.join(val_dir, "normal")))

print("Number of normal images in train +test+val directory:", train_normal_count +test_normal_count+val_normal_count)
print("Number of normal images in test directory:", test_normal_count)
# %%
