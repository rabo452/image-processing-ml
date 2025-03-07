# the script which splits the dataset on equal partitions

import os
import shutil

path = '/home/dimka/dataset2'
dataset_path = '/home/dimka/dataset'

def split_array(arr, n):
    avg_len = len(arr) // n
    remainder = len(arr) % n
    
    result = []
    start = 0
    for i in range(n):
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(arr[start:end])
        start = end
    
    return result

directories = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
n = 16
split_dirs = split_array(directories, n)

for i in range(n):
    copy_dirs = split_dirs[i]
    for copy_dir in copy_dirs:
        destination = os.path.join(path, f'dataset{i + 1}', os.path.basename(copy_dir))
        if not os.path.exists(destination):  # Only copy if the destination does not exist
            shutil.copytree(copy_dir, destination)
        else:
            print(f"Directory {destination} already exists, skipping...")
