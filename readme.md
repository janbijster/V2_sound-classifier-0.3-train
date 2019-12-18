# V2 Sound Classifier v0.3 - Training
This repository is for retraining the sound classifier on your own computer.

The steps below walk you through setting up a python environment, collecting data and training the model.

After this you can evaluate the model's performance or use it to detect recorded sounds in realtime.

For using your (or our) trained model on a raspberry pi, see the repository ...

## 1. Installing python environment
For more experienced python users: the required packages are listed in `requirements.txt`. For users new to Python, read on:

### 1.1 MacOS / Linux
#### 1.1.1 Python3
Download python3 from [here](https://www.python.org/downloads/) if it is not installed.

#### 1.1.2 Dependencies
It's usually a good idea to install the dependencies in a `venv` (short for virtual environmont). This will allow you to install different versions of packages on the same machine. To create a virtual environment:
```
python3 -m venv /path/to/new/virtual/environment
```

Then activate it:
```
source /path/to/new/virtual/environment/bin/activate
```

Then we install the dependencies with pip:
```
pip install numpy scipy llvmlite scikit-image scikit-learn sounddevice tensorflow==1.13.1 Keras==2.2.4 librosa matplotlib Pillow
```

### 1.2 Windows
#### 1.2.1 Python3
Some of the python dependencies such as numpy have to be compiled. Setting up a compiler on Windows can be tricky, so the easiest way to install the python environment on Windows is to use Anaconda. Anaconda allows you to install precompiled binaries from the conda repository.

Download Anaconda Python 3.x distribution from [here](https://www.anaconda.com/distribution/) and install. During the installation you will be asked if you want to add this python interpreter to PATH. Choosing yes will make sure if you type `python` in a command window the Anaconda version will be used, this is the easy option. If you don't want te make the Anaconda python the default one and choose no, it's best to perform the following steps in an *Anaconda prompt* instead of the normal command or powershell window.

#### 1.2.2 Dependencies
Install these dependencies with conda:
```
conda install numpy scipy llvmlite
```

The rest can be installed with pip. But we first install pip inside conda to make sure pip and conda find the same installed packages:
```
conda install pip
```

Then check if the default pip is the Anaconda one:
```
pip -V
```
In the path in the output (e.g. `pip 19.2.3 from D:\Anaconda3\lib\site-packages\pip (python 3.7)`) you can verify if this pip points to the one installed in the Anaconda environment.

Then install the remaining packages:
```
pip install scikit-image scikit-learn sounddevice tensorflow==1.13.1 Keras==2.2.4 librosa matplotlib Pillow
```

Note: if you want to use your GPU, you have to install a number of things like CUDA and cuDNN. See [this page](https://www.tensorflow.org/install/gpu) information how to set up tensorflow with gpu support. You then replace `tensorflow==1.13.1` with `tensorflow-gpu==1.13.1` in the command above. Pay special attention to the versions when installing CUDA and cuDNN, different tensorflow versions ar ecompatible with different CUDA versions. [This list](https://www.tensorflow.org/install/source#gpu) is very helpful for that.

## 2. Data
### 2.1 Available datasets
The best dataset we have been able to find is the [Urban Sounds 8k dataset](https://urbansounddataset.weebly.com/urbansound8k.html). This is compiled from sound recordings uploaded to [freesound.org](https://freesound.org). It consists of 8732 sounds across ten categories which the [SONYC](https://wp.nyu.edu/sonyc) project deem the most important in their research.

The entire dataset can be downloaded [here](https://urbansounddataset.weebly.com/download-urbansound8k.html) (you have to fill in a form) or [here](https://zenodo.org/record/1203745#.XSWOb5MzaL4) without inputting any data.

Note that the SONYC project is compiling an even better dataset from captured audio, and have an active citizen-tagging project running now to classify the sounds. [See here](https://www.zooniverse.org/projects/anaelisa24/sounds-of-new-york-city-sonyc) for info. 

### 2.2 Preprocessing: reordering
The 'Urban sounds 8k' / 'sonyc8k' dataset is organized in folds for training, not by category. To reorganize the data in a folder structure with the classnames as directories, use the script `reorganize_sonyc8k.py`:
```
python3 reorganize_sonyc8k.py -i <path to sonyck8k dataset> -m <path to the dataset's metadata file> -o <output data folder>
```
*Note: run `python3 reorganize_sonyc8k.py --help` for more information about the command line arguments*

### 2.3 Preprocessing: spectograms
The script `load_data.py` reads all the .wav files from the dataset and creates spectogram images. It will create the same folder structure in the output directory as in the input directory.

Usage:
```
python3 load_data.py -d <audio data folder> -m <metadata filepath> -o <spectrogram folder>
```
With
 * `<audio data folder>` the path to the folder containing audio organized in class directories as created by the previous step,
 * `<metadata filepath>` the filepath where to store a new metadata file used by this script, to allow (among others) stopping and continuation of the script and
 * `<sprectrogram folder>` the path where to create the folders with spectrograms.

### 2.4 Split data in train and validation sets
To assess the models classification (and in particular generalization) capabilities we need to reserve a subset of the data, so that we can test it on data that the model has never seen before. So we need a 'validation' folder with spectrograms not in the training set, organized in the same way in class folders. If you use the sonyk8k dataset and have preprocessed it with the scripts above, you can use the following script to split the data:
```
python3 split_data.py -d <spectrogram data folder> -v <validation data folder> -f <fold name>
```
With
 * `<spectrogram data folder>` the path to the folder containing spectrograms organized in class directories as created by the previous step,
 * `<validation data folder>` the filepath where to move the validation samples to
 * `<fold name>` the name of the fold. Default: `fold10`.

*Note: The `reorganize_sonyc8k` script stores the fold name in the samples' filename and this script uses that.*

### 2.5 New data
To train with your own data or to add a category to the model, see ...

## 3. Training
To train the model run:
```
python3 train.py -t <training data folder> -v <validation data folder> -m <model folder> -n <model name>
```

## 4. Evaluation
To evaluate the model run:
```
python3 train.py -t <training data folder> -v <validation data folder> -m <model folder> -n <model name>
```