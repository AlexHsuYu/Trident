# Trident
# Intelligent Manufacturing Big Data Contest 2019

Predict the classification of the characteristic curve based on the temperature of the hot pressing furnace.

cooperate with [@Michael-Liao](https://github.com/https://github.com/Michael-Liao)

## Installation

### go to the project directory install by setup.py
```bash
cd trident
pip install .
```
### you can also run inside docker
```bash
cd trident
docker build -t trident -f Dockerfile . # build the image
```

## Usage

### running inside docker
```bash
docker run -it --rm trident [train/test] # run train test

# save the file to a specific location
docker run -it --rm -v /abs/path/on/host/:/log trident test --save-path /log
# i.e.
docker run -it --rm -v $PWD/assets/results:/log trident test --save-path /log
```
### run directly as a module
```bash
cd trident
python -m trident [train/test]
```
### paths inside docker container
* `/app/data/stage1_train`: where the training data lives
* `/app/data/stage1_test`: where the test data lives
* `/log`: resulting csv file

## Development
we use `pipenv` as our virtual environment 
all the require packages are in the Pipfile

