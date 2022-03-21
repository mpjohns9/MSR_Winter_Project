# Machine Learning for Classifying Baseball Grips

## Overview
A baseball pitcher places high importance on being able to deceive a hitter.
The interplay between pitcher and hitter is not unlike that of two chess 
opponents. As such, if a hitter can detect patterns in a pitcher's behavior
that tip off what pitch is being thrown, the chances of success (getting a hit)
are much higher.  

With this in mind, this project is the preliminary step in attempting to 
assess a pitcher's ability to disguise pitches. Using a convolutional 
neural network (CNN), baseball grips are classified with their corresponding
pitch in real-time. Combined with pose estimation, this could serve as a
foundation for future development on pitch prediction in real-time.

As the following sections describe, one challenge of this project was 
availability to data. Since there aren't any open source datasets for
baseball grip images or pitching videos from a batter's perspective, 
I built my own. This required a meticulous data augmentation process
to generate the synthetic dataset that my model is trained and tested 
on. The details of this process are explored in depth later on in this
README.

Thank you for your time and interest -- enjoy!

## Contents
This package contains the following files. In-depth descriptions for each 
can be found in subsequent sections.  

 - `image_augmentation.py`: generates large synthetic dataset from subset of example images  
 - `image_collection.py`: script used to collect images quickly
 - `main.py`: runs model training and prediction program with modifiable arguments
 - `predict.py`: real-time classification using trained model
 - `train_model.py`: trains CNN on train & test images (examples provided)

 