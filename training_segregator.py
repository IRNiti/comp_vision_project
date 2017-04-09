'''
Team Valuca:
Vasundhara Gupta
Raluca Niti

This is a script to segregate the provided labeled images into the following directory structure based on their images.

/train
    /cats
    /dogs
/eval
    /cats
    /dogs

From the original 6k labelled images, we allocate roughly 75% to a training set and 25% to a labelled eval set (separate from the final, initially unlabelled, test set).

We decided on this split after reading http://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
'''

import os
import shutil

LABELS_CSV = 'Y_Train.csv'
ORIGINAL_COMBINED_TRAIN_DIR = 'X_Train'

NEW_ROOT_TRAIN_DIR = 'segregated_train'
NEW_ROOT_EVAL_DIR = 'segregated_eval'

CATS_DIR = 'cats'
DOGS_DIR = 'dogs'


def ensure_dest_directories_exist():
    if not os.path.exists(NEW_ROOT_TRAIN_DIR):
        os.makedirs(NEW_ROOT_TRAIN_DIR)
    if not os.path.exists(NEW_ROOT_EVAL_DIR):
        os.makedirs(NEW_ROOT_EVAL_DIR)

    train_cat_dir = os.path.join(NEW_ROOT_TRAIN_DIR, CATS_DIR)
    if not os.path.exists(train_cat_dir):
        os.makedirs(train_cat_dir)

    train_dog_dir = os.path.join(NEW_ROOT_TRAIN_DIR, DOGS_DIR)
    if not os.path.exists(train_dog_dir):
        os.makedirs(train_dog_dir)

    eval_cat_dir = os.path.join(NEW_ROOT_EVAL_DIR, CATS_DIR)
    if not os.path.exists(eval_cat_dir):
        os.makedirs(eval_cat_dir)

    eval_dog_dir = os.path.join(NEW_ROOT_EVAL_DIR, DOGS_DIR)
    if not os.path.exists(eval_dog_dir):
        os.makedirs(eval_dog_dir)


if __name__ == '__main__':
    ensure_dest_directories_exist()

    f = open(LABELS_CSV, 'r')
    f.readline()  # skip first line

    num_cats_so_far = 0
    num_dogs_so_far = 0

    for line in f:
        image_filename, label = tuple(line.split(','))
        orig_absolute_path = os.path.join(ORIGINAL_COMBINED_TRAIN_DIR,
                                          image_filename)

        dest_absolute_path = ''
        if int(label) == 0:  # cat
            dest_absolute_path = os.path.join(NEW_ROOT_EVAL_DIR
                                              if num_cats_so_far % 4 == 0 else
                                              NEW_ROOT_TRAIN_DIR, CATS_DIR,
                                              image_filename)
            num_cats_so_far += 1
        else:  # dog
            dest_absolute_path = os.path.join(NEW_ROOT_EVAL_DIR
                                              if num_dogs_so_far % 4 == 0 else
                                              NEW_ROOT_TRAIN_DIR, DOGS_DIR,
                                              image_filename)
            num_dogs_so_far += 1

        try:
            shutil.copyfile(orig_absolute_path, dest_absolute_path)
        except IOError as e:
            print(e)  # shouldn't happen
