import os

# Set the base path based on your image
base_path = r'3d\BraTS\BraTS2020_training_data'

# Check if the path exists to avoid errors
if os.path.exists(base_path):
    print(f"Contents of {base_path}:")
    
    # List all files and subdirectories
    content_list = os.listdir(base_path)
    
    for item in content_list:
        print(f"- {item}")
        
    print(f"\nTotal items found: {len(content_list)}")
else:
    print("The specified path does not exist. Please check the directory string.")