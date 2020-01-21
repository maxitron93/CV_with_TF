# Modern Neural Networks
## Convolutional Neural Networks
CNNs were introduced to solve some of the shortcomings of the original neural networks. In this section, we will address these issues and present how CNNs deal with them.

#### Problems with fully connected networks
We have already highlighted the following two main drawbacks of basic networks when dealing with images:
- <strong>An explosive number of parameters</strong>: Images are complex structures with a large number of values. Even the small, single-channel images of the MINST dataset represent input vectors of 784 values each. For the first layer of the basic neural network we implemented, this meant a weight matrix of shape (784, 64). This equates to 50,176 (784 × 64) parameter values to optimize, just for this variable! This number of parameters simply explodes when we consider larger RGB images or deeper networks.

- <strong>A lack of spatial reasoning</strong>: Because their neurons receive all the values from the previous layer without any distinction (they are fully connected), these neural networks do not have a notion of distance/spatiality. Spatial relations in the data are lost. More precisely, this means that the notion of proximity between pixels is lost to fully connected (FC) layers, as all pixel values are combined by the layers with no regard for their original positions. Intuitively, neural layers would be much smarter if they could take into account spatial information.

#### Introducing CNNs
First of all, CNNs can handle multidimensional data. For images, a CNN takes as input three-dimensional data (height × width × depth) and has its own neurons arranged in a similar volume. 

![CNN Diagram](../assets/CNN_diagram.PNG)

This leads to the second novelty of CNNs—unlike fully connected networks, where neurons are connected to all elements from the previous layer, each neuron in CNNs only has access to some elements in the neighboring region of the previous layer. This region is called the receptive field of the neurons (or the filter size).

By linking neurons only to their neighboring ones in the previous layer, CNNs not only drastically reduce the number of parameters to train, but also preserve the localization of image features.

#### Convolution layers
In these layers, the number of parameters is further reduced by sharing the same weights and bias among all neurons connected to the same output channel. 

These specific neurons with shared weights and bias can also be thought of as a single neuron sliding over the whole input matrix with spatially limited connectivity. At each step, this neuron is only spatially connected to the local region in the input volume (H × W × D) it is currently sliding over. Given this limited input of dimensions, the neuron still works like the ones modeled in our first chapter—it linearly combines the input values before applying an activation function to the sum (a linear or non-linear function).

Therefore, the results of a convolution are indeed obtained by sliding a filter, w, over the input matrix, x, and computing, at each position, the dot product of the filter and the patch of x starting at the current position.

![Concolution Calculation](../assets/convolution_calculation.PNG)

Note: the input, x, has been padded with zeros, which is commonly done in convolutional layers.

A convolutional layer with N sets of different neurons is thus defined by N weight matrices (also called filters or kernels) of shape D × k × k (when the filters are square), and N bias values. Therefore, this layer only has N × (Dk2 + 1) values to train - significantly less tan the number of trainable parameters of fully connected layers. i.e. FC layers are influenced by the dimensionality of the data, whereas CNNs are not. 

This property makes convolutional layers really powerful tools in computer vision for two reasons:
- We can train networks for larger input images without impacting the number of parameters we would need to tune. 
- Convolutional layers can be applied to any images, irrespective of their dimensions! Unlike networks with fully connected layers, purely convolutional ones do not need to be adapted and retrained for inputs of different sizes. 

Other interesting properties of convolutional layers include:
- The layer's filters become really good at reacting to specific local features (a layer with N filters means the possibility to react to N different features). Each kernel of the first convolutional layer in a CNN would, for instance, learn to activate for a specific low-level feature, such as a specific line orientation or color gradient. Then, deeper layers would use these results to localize more abstract/advanced features, such as the shape of a face, and the contours of a particular object. 
- Each filter would respond to a specific image feature, whatever its location(s) in the image. More formally, convolutional layers are invariant to translation in the image coordinate space.

Concolutional layer hyperparameters:
- Number of filters, N
- Input depth, D
- Filter/kernel size (kH, kW). As square filters are commonly used, the size is usually simply defined by k.
- Stride: Defines whether the dot product between the image patches and the filters should be computed at every position when sliding (stride = 1), or every s position (stride = s). The larger the stride, the sparser the resulting feature maps.
- Padding: Images can also be zero-padded before convolution; that is, their sizes can be synthetically increased by adding rows and columns of zeros around their original content.

All these parameters not only affect the layer's operations, but also its output shape. As such, with these hyperparameters, we can easily control the layer's output sizes. This is particularly convenient for applications such as object segmentation; that is, when we want the output segmentation mask to be the same size as the input image.

the Keras API encapsulates the initialization of CNNs in the <em>tf.keras.layers</em> module. We can instantiate a similar convolutional layer in a single call, as follows:

![Concolutional layer](../assets/convolutional_layer.PNG)

The documentation has a long list of additional parameters, encapsulating several concepts, such as weight regularization.

#### Pooling layers 
These pooling layers are a bit peculiar because they do not have any trainable parameters. Each neuron simply takes the values in its window (the receptive field) and returns a single output, computed from a predefined function.

The two most common pooling methods are max-pooling and average-pooling. <em>Max-pooling</em> layers return only the maximum value at each depth of the pooled area, and <em>average-pooling</em> layers compute the average at each depth of the pooled area.

#### Fully connected layers
FC layers are also used in CNNs, the same way they are in regular networks.



