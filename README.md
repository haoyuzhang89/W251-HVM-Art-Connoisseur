# Final Project W251 Spring 2021

### Haoyu Zhang, Yixun Zhang, Patrick Kim

# Training a Neural Network to Classify Art by Artist

## Dataset

Our dataset is a Kaggle dataset available at https://www.kaggle.com/ikarus777/best-artworks-of-all-time. This dataset includes images with artist names as file names and a csv file which has artist names and the properties of the images. We also correct some special characters of the artist names to standard charaters. The dataset contains labeled images from 50 prominent artists which we will divide into a training and test set for our training.

## Goal

The goal of our project is to categorize the images based on the painting styles of the artists. We are using 80% of the dataset as our training data

## Equipment / Data pipeline

We are using a g4dn.2xlarge instance from AWS to train our neural network. The AMI we are using is "Deep Learning AMI (Ubuntu 18.04) Version 42.1." We use the preinstalled tensorflow option in this AMI and use a Jupyter notebook to conduct our training.

## Model

After looking at a few models, we decided to use Resnet-50 as the architecture for our image recognition model. There are a few good reasons to try Resnet in our work:
* Resnet has achieved incredible accuracy due to its clever workaround to reduce the occurence of vanishing gradients when adding more layers to the neural network. This enables neural networks that implement this workaround to have many more layers than most others.
* The model is readily available via Keras
* The training time seems acceptable on our machine for this model

## Difficulties
[Large image data set: resize from 256 to 128]

##



