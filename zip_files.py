folders = ['P016_balloon1_side', 'P016_balloon1_top', 'P016_balloon2_side', 'P016_balloon2_top']

side_folders = []
up_folders = []
for fo in folders:
    if 'side' in fo:
        side_folders.append(fo)
    elif 'top' in fo:
        up_folders.append(fo)
ordered_folders = side_folders + up_folders

import os
from zipfile import ZipFile

# files_address = '/home/user/datasets/frames'
files_address = '/datashare/APAS/frames/'
# zip_address = '/home/user/PycharmProjects/DS.ComputerVisionProject'
zip_address = '/home/student/computer_vision/tomer_final/compressed'

for fo in side_folders:
# Create a ZipFile Object
    with ZipFile(f'{zip_address}/{fo}.zip', 'w') as zip_object:
        for folder_name, sub_folders, file_names in os.walk(f'{files_address}/{fo}'):
            for filename in file_names:
                # Create filepath of files in directory
                file_path = os.path.join(folder_name, filename)
                # Add files to zip file
                zip_object.write(file_path, os.path.basename(file_path))

