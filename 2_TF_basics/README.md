# TensorFlow Basics and Training a Model
-  While you can perform any numerical operation with TensorFlow, it is mostly used to train and run deep neural networks.
- This chapter will introduce you to the core concepts of TensorFlow 2 and walk you through a simple example.

## TensorFlow APIs
TensorFlow's architecture has several levels of abstraction.

![TensorFlow Layers](../assets/tensorflow_layers.PNG)

Let's first introduce the lowest layer and find our way to the uppermost layer.

- Most deep learning computations are coded in C++. To run operations on the GPU, TensorFlow uses a library developed by NVIDIA called CUDA. This is the reason you need to install CUDA if you want to exploit GPU capabilities and why you cannot use GPUs from another hardware manufacturer.

- The Python low-level API then wraps the C++ sources. When you call a Python method in TensorFlow, it usually invokes C++ code behind the scenes. This wrapper layer allows users to work more quickly because Python is considered easier to use than C++ and does not require compilation. This Python wrapper makes it possible to perform extremely basic operations such as matrix multiplication and addition.

- At the top sits the high-level API, made of two components—Keras and the Estimator API. Keras is a user-friendly, modular, and extensible wrapper for TensorFlow. We will introduce it in the next section. The Estimator API contains several pre-made components that allow you to build your machine learning model easily. You can consider them building blocks or templates.

## Introducing Keras
- First released in 2015, Keras was designed as an interface to enable fast experimentation with neural networks. As such, it relied on TensorFlow or Theano to run deep learning operations.

- Since 2017, TensorFlow has integrated Keras fully, meaning that you can use it without installing anything other than TensorFlow. Throughout this book, we will rely on tf.keras instead of the standalone version of Keras.

- We can now move on to building the actual model. We will use a very simple architecture composed of two fully connected (also called dense) layers:

    - <strong>Flatten</strong>: This will take the 2D matrix representing the image pixels and turn it into a 1D array. We need to do this before adding a fully connected layer. The 28 × 28 images are turned into a vector of size 784.
    - <strong>Dense</strong> of size 128: This will turn the 784 pixel values into 128 activations using a weight matrix of size 128 × 784 and a bias matrix of size 128. In total, this means 100,480 parameters.
    - <strong>Dense</strong> of size 10: This will turn the 128 activations into our final prediction. Notice that because we want probabilities to sum to 1, we will use the softmax activation function.

## Tensorflow 2 and Keras in Detail
We have introduced the general architecture of TensorFlow and trained our first model using Keras. Let's now walk through the main concepts of TensorFlow 2.

