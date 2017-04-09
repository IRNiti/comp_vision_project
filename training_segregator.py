'''
Team Valuca:
Vasundhara Gupta
Raluca Niti

This is a script to segregate the provided labeled images into the following directory structure based on their images.

/train
    /cats
    /dogs


To work with the Pytorch ImageFolder API, it also randomly segregates the test images, though these labels are random and mostly meaningless.

/test
    /cats
    /dogs
'''

import os
import shutil

LABELS_CSV = 'Y_Train.csv'
ORIGINAL_COMBINED_TRAIN_DIR = 'X_Train'
ORIGINAL_COMBINED_TEST_DIR = 'X_Test'

NEW_ROOT_TRAIN_DIR = 'segregated_train'
NEW_ROOT_TEST_DIR = 'rand_segregated_test'

CATS_TRAIN_DIR = os.path.join(NEW_ROOT_TRAIN_DIR, 'cats')
DOGS_TRAIN_DIR = os.path.join(NEW_ROOT_TRAIN_DIR, 'dogs')

CATS_TEST_DIR = os.path.join(NEW_ROOT_TEST_DIR, 'cats')
DOGS_TEST_DIR = os.path.join(NEW_ROOT_TEST_DIR, 'dogs')


def ensure_dest_directories_exist():
    if not os.path.exists(NEW_ROOT_TRAIN_DIR):
        os.makedirs(NEW_ROOT_TRAIN_DIR)
    if not os.path.exists(CATS_TRAIN_DIR):
        os.makedirs(CATS_TRAIN_DIR)
    if not os.path.exists(DOGS_TRAIN_DIR):
        os.makedirs(DOGS_TRAIN_DIR)

    if not os.path.exists(NEW_ROOT_TEST_DIR):
        os.makedirs(NEW_ROOT_TEST_DIR)
    if not os.path.exists(CATS_TEST_DIR):
        os.makedirs(CATS_TEST_DIR)
    if not os.path.exists(DOGS_TEST_DIR):
        os.makedirs(DOGS_TEST_DIR)


if __name__ == '__main__':
    ensure_dest_directories_exist()

    f = open(LABELS_CSV, 'r')
    f.readline()  # skip first line

    for line in f:
        image_filename, label = tuple(line.split(','))
        orig_absolute_path = os.path.join(ORIGINAL_COMBINED_TRAIN_DIR,
                                          image_filename)

        dest_absolute_path = os.path.join(CATS_TRAIN_DIR, image_filename) if int(label) == 0 else os.path.join(DOGS_TRAIN_DIR, image_filename)

        try:
            shutil.copyfile(orig_absolute_path, dest_absolute_path)
        except IOError as e:
            print(e)  # shouldn't happen

    for i, image_filename in enumerate(os.listdir(ORIGINAL_COMBINED_TEST_DIR)):
        src_absolute_path = os.path.join(ORIGINAL_COMBINED_TEST_DIR, image_filename)
        
        dest_absolute_path = os.path.join(CATS_TEST_DIR, image_filename) if i % 2 == 0 else os.path.join(DOGS_TEST_DIR, image_filename)

        shutil.copyfile(src_absolute_path, dest_absolute_path)
