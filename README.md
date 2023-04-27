# A Holistic Approach for Aligned Music Notation and Lyrics Transcription

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
```

### Docker
If you are using Docker to run experiments, create an image with the provided Dockerfile:

```sh
docker build -t <your_tag> .
docker run -itd --rm --gpus all --shm-size=8gb -v <repository_path>:/workspace/ <image_tag>
docker exec -it <docker_container_id> /bin/bash
```

# Data

The datasets created to run the experiments are publicly available for replication purposes. 

**Download and setup**

```sh
cd Data
wget <link_to_datasets>
tar -xzvf ICDAR2023_AMLT_Datasets.tgz
```

## Including new datasets


---

**Citation**

Citation information will be provided when proceedings are published.

----
