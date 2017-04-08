'''
Vasundhara Gupta
Raluca Niti

This is a script to segregate the files into the following directory structure:

/train
    /cats
    /dogs
'''

import os
import shutil

LABELS_CSV = 'Y_Train.csv'
ORIGINAL_COMBINED_TRAIN_DIR = 'X_Train'

NEW_ROOT_TRAIN_DIR = 'segregated_train'

CATS_DIR = os.path.join(NEW_ROOT_TRAIN_DIR, 'cats')
DOGS_DIR = os.path.join(NEW_ROOT_TRAIN_DIR, 'dogs')

if __name__ == '__main__':
    f = open(LABELS_CSV, 'r')
    f.readline()  # skip first line

    for line in f:
        image_filename, label = tuple(line.split(','))

        orig_absolute_path = os.path.join(ORIGINAL_COMBINED_TRAIN_DIR, image_filename)
        dest_absolute_path = os.path.join(CATS_DIR if int(label) == 0 else DOGS_DIR, image_filename)
        
        try:
            shutil.copyfile(orig_absolute_path, dest_absolute_path)
        except IOError as e:
            print(e)
