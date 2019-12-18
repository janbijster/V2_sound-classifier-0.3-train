import os
import argparse

# params
default_data_folder_spectrograms = './data/spectrograms/'
default_data_folder_validation = './data/spectrograms-test/'
default_validation_fold = 'fold10'

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default=default_data_folder_spectrograms,
    help='The data directory containing the spectrograms. (default: {})'.format(default_data_folder_spectrograms))
parser.add_argument('-v', '--validation', default=default_data_folder_validation,
    help='The directory where to move the validation data to to. (default: {})'.format(default_data_folder_validation))
parser.add_argument('-f', '--fold', default=default_validation_fold,
    help='The fold to separate . (default: {})'.format(default_validation_fold))
args = parser.parse_args()

# script
if not os.path.exists(args.validation):
	os.makedirs(args.validation)

# loop over categories
for dir_entry in [f for f in os.scandir(args.data) if f.is_dir()]:
    test_dir = os.path.join(args.validation, dir_entry.name)
    if not os.path.exists(test_dir):
	    os.makedirs(test_dir)
    # loop over files
    for file_entry in [f for f in os.scandir(dir_entry.path) if not f.is_dir()]:
        if args.fold in file_entry.name:
            os.rename(file_entry.path, os.path.join(test_dir, file_entry.name)) 