import argparse
from training_functions import train_model

# dimensions of our images.
img_width, img_height = 128, 128

# default arguments
default_train_data_dir = './data/spectrograms/'
default_validation_data_dir = './data/spectrograms-test/'
default_models_folder = './models'
default_model_name = 'model'

# script
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', default=default_train_data_dir,
    help='Training data folder. Default: {}'.format(default_train_data_dir))
parser.add_argument('-v', '--validation', default=default_validation_data_dir,
    help='Validation data folder. Default: {}'.format(default_validation_data_dir))
parser.add_argument('-m', '--models', default=default_models_folder,
    help='Models folder. Default: {}'.format(default_models_folder))
parser.add_argument('-n', '--name', default=default_model_name,
    help='Model name, without ".h5". Default: {}'.format(default_model_name))
args = parser.parse_args()

model = train_model(
    train_data_dir=args.train,
    validation_data_dir=args.validation,
    image_dimensions=(img_width, img_height),
    model_name=args.name,
    models_folder=args.models
)