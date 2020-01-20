# Contents

## 1 Computer Vision and Neural Networks
- Introduction to Computer Vision, including the challenges associated with it, and some historical methods, such as SIFT and SVMs.
- Introduction to Neural Networks, including how they are built, trained and applied.
- Implementation of classifier network from scratch.

## 2 TensorFlow Basic and Training a Model
- Train a basic CV model with using the Keras API.
- Introduced the main concept behind TF2.0, including tensors, graphs, AutoGraph, eager execution, the gradient tape, and other more advanced concepts of the framework.
- Walkthrough of the main tools used to support deep learning in TF, including TensorBoard for monitoring, the TFX for pre-processing and model analysis.
- Overview of where to run your model based on your needs.

## 3 Modern Neural Networks
- Overview of Deep Learning
- Overview of CNNs
- Overview of optimization and regularization techniques, including various optimizers, L1/L2 regularization, dropout, and batch normalization

## 4 Influential Classification Tools
- Overview of several CNN architectures applied to the task of classifying large picture datasets.
- Introduction to transfer learning, and how to reuse state-of-the-art solutions.

## 5 Object Detection Models
- Goes through the architecture of two object detection models; YOLO, known for its inference speed, and Faster R-CNN, known for its state-of-the-art performance. 

## 6 Enhancing and Segmenting Images
- Extends on object detection by learning how to segment images into meaningful part, as well as how to transform and enhance them. 
- Covers several paradigms for pixel-precise applications, including encoders-decoders and other specific architectures. 
- Covers more complex tasks, like semantic segmentation. 

## 7 Training on Complex and Scarce Datasets
- Provides in-depth detail of how TensorFlow can be used to effectively augment and serve training batches.
- Specifically covers how to use <em>tf.data</em> to optimize data flow. 

## 8 Video and Recurrent Neural Networks
- Covers the general principles of RNNs.
- Discusses the limitations of RNNs due to gradient vanishing, and introduces an alternative architecture; LSTM networks.
- Apply CNNs and LSTMs to classify videos. Includes going over video-specific techniques such as frame sampling and padding. 

## 9 Optimizing Models and Deploying on Mobile Devices
- Covers several topics on performance, including how to properly measure inference speed of a model.
- Covers techniques to reduce inference time, including choosing the right hardware and right libraries, optimizing input size, and optimizing post-processing.
- Covers how to make a slower model to appear, to the user, as if it were processing in real time, and to reduce the model size.
- Introduces on-device ML, including how to convert TF and Keras models to a formal that's compatible with on-device DL frameworks.
