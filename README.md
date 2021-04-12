# Final Project W251 Spring 2021

### Haoyu Zhang, Yixun Zhang, Patrick Kim

# Training a Neural Network to Classify Art by Artist

## Goal

Different artists have various painting styles. The goal of our project is to identify the painter of the images based on their painting styles.

## Raw Dataset

Our dataset is a Kaggle dataset available at https://www.kaggle.com/ikarus777/best-artworks-of-all-time. The dataset contains labeled images from 50 prominent artists which we will divide into a training and test set for our training. And the data folder also has a csv file with properties of the images.

## Preprocess Dataset

* We corrected some special characters of the artist names to standard charaters, so our model can read the names.
* We 

## Infrastructure

We are using a g4dn.2xlarge instance from AWS to train our neural network. The AMI we are using is "Deep Learning AMI (Ubuntu 18.04) Version 42.1." We use the preinstalled tensorflow option in this AMI and use a Jupyter notebook to conduct our training.

## Model Framework

After looking at a few models, we decided to use Resnet-50 as the architecture for our image recognition model. There are a few good reasons to try Resnet in our work:
* Resnet has achieved incredible accuracy due to its clever workaround to reduce the occurence of vanishing gradients when adding more layers to the neural network. This enables neural networks that implement this workaround to have many more layers than most others.
* The model is readily available via Keras
* The training time seems acceptable on our machine for this model

## Model Training

## Model Results

## Difficulties
[Large image data set: resize from 256 to 128]


