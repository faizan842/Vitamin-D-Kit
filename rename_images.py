import os

# Path to the directory containing the files
directory_path = '/Users/faizanhabib/Downloads/Main'

# Loop through all the files in the directory
for filename in os.listdir(directory_path):
    # Check if the file ends with .jpg
    if filename.endswith('.jpg'):
        # Create the new filename by replacing .jpg with .JPG and adding 2- at the beginning
        new_filename = '2-' + filename.replace('.jpg', '.JPG')
        # Construct full file path
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} to {new_file}')

print('All files have been renamed.')
