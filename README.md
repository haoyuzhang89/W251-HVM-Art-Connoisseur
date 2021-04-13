# Final Project W251 Spring 2021

### Haoyu Zhang, Yixun Zhang, Patrick Kim

# Training a Neural Network to Classify Art by Artist


## Goal

Different artists have various painting styles. The goal of our project is to identify the painter of the images based on their painting styles.


## Raw Dataset

Our dataset is a Kaggle dataset available at https://www.kaggle.com/ikarus777/best-artworks-of-all-time. The dataset contains 8000+ labeled images from 50 prominent artists which we will divide into a training and test set for our training. And the data folder also has a csv file with artist information, like id of the artist, artist name, living years, genre, nationality and etc.


## Preprocess Dataset

* We corrected some special characters of the artist names to standard charaters, so our model can read the names.
* We converted the dataset to 3 RGB channels by "/255" to the original pixels.
* We split the dataset into 80%  training data, 10% dev data and 10% test data.
* Besides the above modifications, we also created two new datasets from the raw dataset. In the first type of dataset, we sampled the cropped images. We cropped the images with the sizes of 256 * 256, 128 * 128, and 64 * 64 and fed those images to the model for training. Here are some examples of the croped images for random sampling.
  ![image](https://user-images.githubusercontent.com/59550524/114339653-8f3c3d00-9b0a-11eb-881d-aad6af7783df.png)
* The other dataset generated from the raw dataset is the reshaping dataset. We reshaped the whole images to the same sizes. And fed those reshaped images to the training model.
  ![image](https://user-images.githubusercontent.com/59550524/114339764-d6c2c900-9b0a-11eb-9843-0c755f9f48d1.png)


## Infrastructure

We are using a g4dn.2xlarge instance from AWS to train our neural network. The AMI we are using is "Deep Learning AMI (Ubuntu 18.04) Version 42.1." We use the preinstalled tensorflow option in this AMI and use a Jupyter notebook to conduct our training.


## Model Frameworks

After researching and experimenting Kera models, we chose to do further training on the ResNet50 and MobileNet architecture. As the available model table shows below, both ResNet50 and MobileNet have model size small enough for us to train meanwhile they have decent accuracy.
  ![image](https://user-images.githubusercontent.com/59550524/114341921-b8ab9780-9b0f-11eb-91a5-363e1b43ecc3.png)[1]

ResNet50 is a variant of ResNet(Residual Network) model which is a deep convolutional neural network. "Several layers are stacked and are trained to the task at hand."[2] And at the end of its layers, ResNet learns residual. Residual is subtraction of feature learned from input of that layer. ResNet50 has 50 Convolution layers along with 1 MaxPool and 1 Average Pool layer. Here is an architecture of ResNet50:
  ![image](https://user-images.githubusercontent.com/59550524/114344907-9288f600-9b15-11eb-9dab-533728415455.png)[4]

MobileNet is a lightweight deep neural network. It has a high classification accuracy when considering its low number of parameters. "MobileNet is a streamlined architecture that uses depthwise separable convolutions to construct lightweight deep convolutional neural network."[3] Here is an architecture of MobileNet.
  ![image](https://user-images.githubusercontent.com/59550524/114344151-28238600-9b14-11eb-9c40-07f4d8becb9a.png)[3]


## Model Training

After selecting the ResNet50 and MobileNet, we tuned parameters for those two frameworks. Each training took us about 3~5 hours. The "Image size" column in the following table shows the size of the input images. For instance, if the image size is "64", it means the image size is 64 * 64. The sampling size means how many parts of the input image is sampled into the training model. "Model" indicates that the training model is MobilNet or ResNet50. "Resize Or Reshape" describes the preprocessing of the origin images. "Epoch" is the epoch of the training model. "Dropout Rate" indicates the percentage of data points are dropped out from the inputs. "L2" is the shrink weights for L2 regularisation. The rest of the columns are the accuracies from different training or testing stages or metrics. Here is a summary table of the settings and results:

Image Size | Sampling Sizes | Model | Resize Or Reshape | Epoch | Dropout Rate | L2 | Training Accuracy | Top5 Training Accuracy | Validation Accuracy | Top5 Validation Accuracy | Test Accuracy | Top5 Test Accuracy
--- | --- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
64 | 16 | MobileNet | Resize | 100 | 0.7 | Null | 0.7563 |	0.9563 | 0.2416 |	0.5579 | 0.2614 | 0.5738
64 | 16 | MobileNet | Resize | 100 | 0.7 | 0.001 | 0.7563 | 0.9563 | 0.2416 | 0.5579 |	0.2416 | 0.5579
64 | 16 | MobileNet | Resize | 100 | 0.9 | 0.0001 | 0.7429 |	0.9394 | 0.2459 |	0.5457 | 0.2459 | 0.5457
64 | 16 | ResNet50 | Resize | 100 | 0.9 | 0.0001 | 0.8548 | 0.9723 |	0.3157 | 0.6402 |	0.3157 | 0.6402
128 | 4 | MobileNet | Resize | 100 | 0.7 | 0.0001 | 0.8915 |	0.9896 | 0.2264	| 0.5297 | 0.2264 | 0.5297
128 | 4 | ResNet50 | Resize | 100 | 0.7 | 0.0001 | 0.9570 | 0.9972 |	0.2009 | 0.4936 |	0.2009 | 0.4936 
256 | 1 | MobileNet | Resize | 100 | 0.7 | 0.0001 | 0.9486 | 0.9973 |	0.1811 | 0.4675 |	0.1811 | 0.4675
256 | 1 | MobileNet | Reshape | 100 | 0.7 | 0.0001 |	0.9768 | 0.9997 |	0.3598 | 0.6521 | 0.3598 | 0.6521 
256 | 1 | ResNet50 | Resize | 100 | 0.9 | 0.01 | 0.8874 | 0.9885 | 0.1669 | 0.4355 | 0.1669 | 0.4355
256 | 1 | ResNet50 | Reshape | 100 | 0.9 | 0.01 | 0.9232 | 0.9963	| 0.3041 | 0.6012 | 0.3041 | 0.6012
256 | 1 | ResNet50 | Reshape | 100 | 0.9 | 0.0001 | 0.9642 | 0.9984 | 0.3432 | 0.6675 | 0.3432 | 0.6675



## Model Results

Here is the table of details about the top three accurate models we acquired in this project. 

Model Id | Image Size | Sampling Sizes | Model | Resize Or Reshape | Epoch | Dropout Rate | L2 | Training Accuracy | Top5 Training Accuracy | Validation Accuracy | Top5 Validation Accuracy | Test Accuracy | Top5 Test Accuracy
--- | --- | --- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
ModelA | 64 | 16 | ResNet50 | Resize | 100 | 0.9 | 0.0001 | 0.8548 | 0.9723 |	0.3157 | 0.6402 |	0.3157 | 0.6402
ModelB | 256 | 1 | MobileNet | Reshape | 100 | 0.7 | 0.0001 | 0.9768 | 0.9997 |	0.3598 | 0.6521 | 0.3598 | 0.6521
ModelC | 256 | 1 | ResNet50 | Reshape | 100 | 0.9 | 0.0001 | 0.9642 | 0.9984 | 0.3432 | 0.6675 | 0.3432 | 0.6675 

And here are the plots of the model performance and model loss of those top 3 models.
The plots of ModelA:
![image](https://user-images.githubusercontent.com/59550524/114500980-f1b23d80-9bdd-11eb-95f1-49332283deeb.png)

The plots of ModelB:
![image](https://user-images.githubusercontent.com/59550524/114500615-3db0b280-9bdd-11eb-96af-d3836e019534.png)

The plots of ModelC:
![image](https://user-images.githubusercontent.com/59550524/114501409-cb40d200-9bde-11eb-977f-645392e16ab9.png)

Overall, it seems like the reshaped data performs better than the cropped data. Interestingly, we got our best results out of MobileNet reshaped data with a 36% top 1 test accuracy and a 65% top 5 test accuracy.

Following are some model prediction examples,

* Andy Warhol:

<p align="center">
  <img width="460" height="300" src="https://github.com/pakim249CAL/W251-FinalProject/blob/master/demo/model2c/warhol2c.png">
</p>

* Claude Monet

<p align="center">
  <img width="383" height="460" src="https://github.com/pakim249CAL/W251-FinalProject/blob/master/demo/model4b/monet4b.png">
</p>

* Édouard Manet

<p align="center">
  <img width="460"  src="https://github.com/pakim249CAL/W251-FinalProject/blob/master/demo/model2c/manet2c.png">
</p>



## Difficulties

* We ran into overfitting when we were training the data. When loss function of the training model went down, the loss function of the validation model went up a lot. So, we introduced the dropout parameter and L2 regularisation. After tuning hyperparameters (dropout rate) and introducing regularization (l2), although this problem couldn’t be totally resolved with acquiring a reasonable accuracy at the same time, the overfitting issues are not as bad as before for most of the models. 
* We also hit large scale of data issues. The model kept crashing when the training data was too large. We attempted to fix this by trying out reduced image sizes. One solution we considered was to have the training data read in batches from the hard drive instead of holding all the training data in memory, but we didn't get around to implementing this solution.
* There is also a highly uneven distribution of artworks per artist. Some artists have many samples to throw into the network, but others have very few. The lowest number of paintings available for training a particular artist is 24.


## Future Wishlist

* If we have more time and more training resources, we would like to do more training and testing with gray scales and compare the result with colorful images.
* An analysis of scores besides accuracy is probably necessary since it's likely that only a few classes are trained well.


## Reference

[1] https://keras.io/api/applications/

[2] https://iq.opengenus.org/resnet50-architecture/#:~:text=ResNet50%20is%20a%20variant%20of,explored%20ResNet50%20architecture%20in%20depth.

[3] https://www.hindawi.com/journals/misy/2020/7602384/

[4] https://link.springer.com/article/10.1007/s00330-019-06318-1

