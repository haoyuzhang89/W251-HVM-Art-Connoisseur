# Final Project W251 Spring 2021

### Haoyu Zhang, Yixun Zhang, Patrick Kim

# Training a Neural Network to Classify Art by Artist

## Goal

Different artists have various painting styles. The goal of our project is to identify the painter of the images based on their painting styles.

## Raw Dataset

Our dataset is a Kaggle dataset available at https://www.kaggle.com/ikarus777/best-artworks-of-all-time. The dataset contains 8000+ labeled images from 50 prominent artists which we will divide into a training and test set for our training. And the data folder also has a csv file with artist information, like id of the artist, artist name, living years, genre, nationality and etc.

## Preprocess Dataset

* We corrected some special characters of the artist names to standard charaters, so our model can read the names.
* We normalized the dataset by "/255" to the original pixels.
* We splitted the dataset by 80% of training data, 10% of dev data and 10% of test data.
* We created two new datasets from the raw dataset. In the first type of dataset, we sampled the cropped images. We cropped the images with the sizes of 256 * 256, 128 * 128, and 64 * 64. And fed same sizes of images to the model for training.
* ![image](https://user-images.githubusercontent.com/59550524/114339653-8f3c3d00-9b0a-11eb-881d-aad6af7783df.png)

* The other dataset generated from the raw dataset is the reshaping dataset.

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


