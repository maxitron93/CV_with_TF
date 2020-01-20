# 1 Computer Vision and Neural Networks
- Start with a brief presentation of the CV field and its history
- Introduce artificial neural networks and explain how they have revolutionized computer vision
- Implement a neural network from scratch

## 1.1 Computer Vision in the Wild
Computer vision can be summarized as the automated extraction of information from digital images The goal of computer vision is to teach computers how to make sense of these pixels the way humans (and other creatures) do, or even better. 
### Main tasks and their applications
- <strong>Object classification</strong>: The task of assigning proper labels (or classes) to images among a predefined set. Object classification became famous for being the first success story of deep convolutional neural networks being applied to computer vision back in 2012. Common applications are text digitization (using character recognition) and the automatic annotation of image databases. Covered in <strong>Chapter 4</strong>.

![Object classification](../assets/object_classification.PNG)

- <strong>Object identification</strong>: Object identification (or instance classification) methods learn to recognize specific instances of a class. For example, an identification method would focus on the face's features to identify the person and recognize them in other images. Therefore, object identification can be seen as a procedure to cluster a dataset. Covered in <strong>Chapter 6</strong>.

![Object identification](../assets/object_identification.PNG)

- <strong>Object detection and localization</strong>: Another task is the detection of specific elements in an image. It is commonly applied to face detection for surveillance applications or even advanced camera apps, the detection of cancerous cells in medicine, the detection of damaged components in industrial plants, and so on. Detection is often a preliminary step before further computations, providing smaller patches of the image to be analyzed separately. Covered in <strong>Chapter 5</strong>.

![Object detection and localization](../assets/object_detection_and_localization.PNG)

- <strong>Object and instance segmentation</strong>: Segmentation can be seen as a more advanced type of detection. Instead of simply providing bounding boxes for the recognized elements, segmentation methods return masks labeling all the pixels belonging to a specific class or to a specific instance of a class. This makes the task much more complex, and actually one of the few in computer vision where deep neural networks are still far from human performance. Covered in <strong>Chapter 6</strong>.

![Object and instance segmentation](../assets/object_and_instance_segmentation.PNG)

- <strong>Pose estimation</strong>: Pose estimation can have different meanings depending on the targeted tasks. For rigid objects, it usually means the estimation of the objects' positions and orientations relative to the camera in the 3D space. For non-rigid elements, pose estimation can also mean the estimation of the positions of their sub-parts relative to each other.

![Pose estimation](../assets/pose_estimation.PNG)

- <strong>Video analysis</strong>: Computer vision not only applies to single images, but also to videos. If video streams are sometimes analyzed frame by frame, some tasks require that you consider an image sequence as a whole in order to take temporal consistency into account. Covered in <strong>Chapter 8</strong>.

- <strong>Instance tracking</strong>: Tracking is, localizing specific elements in a video stream. Tracking could be done frame by frame by applying detection and identification methods to each frame. However, it is much more efficient to use previous results to model the motion of the instances in order to partially predict their locations in future frames.

- <strong>Action recognition</strong>: Action recognition belongs to the list of tasks that can only be run with a sequence of images. Recognizing an action means recognizing a particular motion among a predefined set. Applications range from surveillance to human-machine interactions.

![Action recognition](../assets/action_recognition.PNG)

- <strong>Motion estimation</strong>: Instead of trying to recognize moving elements, some methods focus on estimating the actual velocity/trajectory that is captured in videos. It is also common to evaluate the motion of the camera itself relative to the represented scene (egomotion). This is particularly useful in the entertainment industry, for example, to capture motion in order to apply visual effects or to overlay 3D information in TV streams such as sports broadcasting.

- <strong>Content-aware image edition</strong>: Besides the analysis of their content, computer vision methods can also be applied to improve the images themselves. More and more, basic image processing tools (such as low-pass filters for image denoising) are being replaced by smarter methods that are able to use prior knowledge of the image content to improve its visual quality. Covered in <strong>Chapters 6 and 7</strong>

![Content aware image edition](../assets/content_aware_image_edition.PNG)

- <strong>Scene reconstruction</strong>: Scene reconstruction is the task of recovering the 3D geometry of a scene, given one or more images. Advanced methods take several images and match their content together in order to obtain a 3D model of the target scene. This can be applied to the 3D scanning of objects, people, buildings, and so on.

## 1.1 A brief history of computer vision
In order to better understand the current stand of the heart and current challenges of computer vision, we suggest that we quickly have a look at where it came from and how it has evolved in the past decades.

