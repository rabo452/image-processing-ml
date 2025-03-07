import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from image_retrieval.train_model import trainModelOnDataset 
from image_retrieval.index_images import indexImages

path = '/home/dimka/new_dataset'

trainModelOnDataset(path)
indexImages(path)
# n = 16
# for i in range(8, n + 1):
#     print(f'training model on dataset {i}...')
#     trainModelOnDataset(f'/home/dimka/dataset2/dataset{i}')
#     print('indexing the images...')
#     indexImages(f'/home/dimka/dataset2/dataset{i}')