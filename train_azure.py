# train_azure.py
import os
import sys
import argparse
from azureml.core import Run, Dataset # Dataset import is not strictly needed here but good practice

# Get the Azure ML run context
run = Run.get_context()

# Add the 'ml' directory to the Python path so we can import modules from it
# This assumes train_azure.py is in the parent directory of 'ml'
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

# Import your main training function from efficientnet_model
from efficientnet_model import main as train_main_function
from efficientnet_model import config as model_config # Import the config from efficientnet_model

if __name__ == '__main__':
    print("Starting Azure ML training job entry script...")

    # Define an argument parser to get the input data path
    parser = argparse.ArgumentParser(description="Pneumonia Detection Training on Azure ML")
    # This 'data_folder' argument will be provided by Azure ML when the dataset is mounted
    parser.add_argument('--data_folder', type=str, help='Path to the mounted dataset folder')
    args = parser.parse_args()

    if args.data_folder:
        # Override the data_dir and metadata_path in efficientnet_model's config
        # to point to the mounted dataset location.
        # args.data_folder will be the path to the *base* directory of your dataset
        # (e.g., /mnt/azureml/datastores/workspaceblobstore/pneumonia-data/)
        # So, meta_data.csv is directly in args.data_folder
        # and train/val/test are subdirectories of args.data_folder
        model_config['data_dir'] = args.data_folder
        model_config['metadata_path'] = os.path.join(args.data_folder, "meta_data.csv")
        print(f"Overriding model config data_dir to: {model_config['data_dir']}")
        print(f"Overriding model config metadata_path to: {model_config['metadata_path']}")
    else:
        print("Warning: --data_folder argument not provided. Using default paths from efficientnet_model.py.")

    # Call your existing main training function
    train_main_function()

    run.complete()
    print("Azure ML training job completed.")

