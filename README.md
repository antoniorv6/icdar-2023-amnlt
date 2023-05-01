# A Holistic Approach for Aligned Music Notation and Lyrics Transcription

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)
[![License](https://img.shields.io/static/v1?label=License&message=MIT&color=blue)]() [![Version](https://img.shields.io/static/v1?label=Version&message=1.0&color=)]() [![Python](https://img.shields.io/static/v1?label=Python&message=3.9&color=blue)]()

This repository contains the public implementation of the paper:

>**Juan C. Martinez-Sevilla**, Antonio Ríos-Vila, Francisco J. Castellanos, Jorge Calvo-Zaragoza<br />
  *[A Holistic Approach for Aligned Music Notation and Lyrics Transcription](https://zenodo.org/record/6573248)*<br />
  17th International Conference on Document Analysis and Recognition, August 21st-26th 2023

Which implements an end-to-end Optical Music Recognition method that outputs the transcription of both the music and lyrics of a given staff-level music score.

In this repository you will find:

- Access links to the datasets created to perform our experiments.
- Source code of the neural network model and carried experiments in the paper.
- Implementation of the music scores synthetic generator used in this paper. 

# Project setup
This implementation has been developed in Python 3.9, PyTorch 2.0 and CUDA 12.0. 

It should work in earlier versions.

To setup a project, run the following configuration instructions:

### Python virtual environment

Create a virtual environment using either virtualenv or conda and run the following:

```sh
git clone https://github.com/antoniorv6/icdar-2023-amnlt.git
pip install -r requirements.txt
mkdir Data
```

### Docker
If you are using Docker to run experiments, create an image with the provided Dockerfile:

```sh
docker build -t <your_tag> .
docker run -itd --rm --gpus all --shm-size=8gb -v <repository_path>:/workspace/ <image_tag>
docker exec -it <docker_container_id> /bin/bash
```

# Data

The datasets created to run the experiments are [publicly available](https://grfia.dlsi.ua.es/musicdocs/ICDAR2023_AMNLT.tgz) for replication purposes. 

**Download and setup**

```sh
cd Data
wget https://grfia.dlsi.ua.es/musicdocs/ICDAR2023_AMNLT.tgz
tar -xzvf ICDAR2023_AMLT_Datasets.tgz
```
**Using the Music Generator**

You can download the implemented to create the datasets of this paper either by downloading it through its [repository link](https://github.com/JuanCarlosMartinezSevilla/ICDAR-23-AMNLT-Music-Generator.git) or by cloning it as a submodule in this repository:

```sh
git submodule update --remote
```
To generate a new dataset, we refer to the [tool docummentation](https://github.com/JuanCarlosMartinezSevilla/ICDAR-23-AMNLT-Music-Generator/blob/main/README.md).

# Train
These experiments run under the Weights & Biases API. To replicate an experiment, run the following code:

```sh
wandb login
python main_train.py --config <path-to-config>
```
The config files are located in the ```config/``` folder, depending on the executed config file, a specific experiment will be run.

# Transcribe a dataset
If you want to use the model to transcribe an unlabeled corpus, you can by running the ``predict_on_dataset.py`` script. To do so, run the following command:

```sh
 python predict_on_dataset.py --images_path <path_to_images> --model <model_name> --checkpoint_path <checkpoint_path> --corpus_name <name_of_the_corpus> --output_path <ouptut_folder_path>
```
The argument parameters are the following:

* ``images_path``: Folder to the images to be transcribed. The tool only supports JPG and PNG images.
* ``model``: Model architecture to load. The following can be inserted:
  * FCN
  * CRNN
  * CNNT_1D
  * CNNT_2D
* ``checkpoint_path``: Folder where the .ckpt file is stored with the weights of the model.
* ``corpus_name``: Name of the corpus, it is essential to be the same name as the dictionaries file in the project.
* ``output_path``: Folder where predictions will be stored.

# Using other datasets

## Train on a new dataset

If you want to test this code with a new music dataset, by folowing these instructions:

1. Insert the new dataset in the ```Data``` folder. It should be divided in three folders (```train```, ```val``` and ```test```). Each folder should contain all the png files and their corresponding Humdrum Kern (.krn) files.

2. Create a configuration file in the ```config``` folder. We recommend copying the format that is provided in the experimentation examples.

3. Run the training command provided in [the training section](#train) including your config file.


# Citation

Citation information will be provided when proceedings are published.

----
