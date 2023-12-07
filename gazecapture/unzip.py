import os
import tarfile

# Set the path to the directory containing the numbered .tar.gz files
directory_path = r'C:\Users\Anja\gaze_data'

# List all files in the directory
files = os.listdir(directory_path)


# Loop through each file in the directory
for file in files:
    # Check if the file is a .tar.gz archive (you may need to adjust the condition)
    if file.endswith('.tar.gz'):
        # Construct the full path to the .tar.gz file
        tar_file_path = os.path.join(directory_path, file)
        
        # Extract the .tar.gz file to the same directory
        with tarfile.open(tar_file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(directory_path)


# Print a message when the extraction is complete
print("Extraction of .tar.gz files is complete.")
