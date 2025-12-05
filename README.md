# BeastVision: Multi-Class Animal Classifier


# Objective

Build a deep learning system that can accurately identify animals in images. The project explores the dataset, implements a convolutional neural network (CNN) for classification, and demonstrates end-to-end ML pipeline best practices.

# Dataset

The dataset contains 1,944 images divided into 15 classes, with each class in its own folder. Images are sized 224 x 224 x 3, suitable for CNN-based image classification.

*Classes:
  
  * Bear
  
  * Bird
  
  * Cat
  
  * Cow
  
  * Deer
  
  * Dog
  
  * Dolphin
  
  * Elephant
  
  * Giraffe
  
  * Horse
  
  * Kangaroo
  
  * Lion
  
  * Panda
  
  * Tiger
  
  * Zebra

# Project Highlights

Developed an end-to-end image classification pipeline using TensorFlow/Keras.

Preprocessed images with resizing, normalization, and one-hot encoding of labels.

Split the dataset into training (80%) and validation (20%), using caching and prefetching for improved performance.

Designed a deep CNN architecture with Conv2D, BatchNormalization, MaxPooling, Dropout, and Dense layers to prevent overfitting.

Applied Min-Max scaling to normalize pixel values.

Visualized data distribution and sample images using Matplotlib and Seaborn.

Trained the model for 120 epochs using the Adam optimizer with categorical crossentropy loss.

Maintained a compact model (~263k parameters) with strong classification performance.

Documented model architecture, summary, and training history for reproducibility.

# Tools & Libraries

Programming Language: Python

Deep Learning: TensorFlow, Keras

Data Handling: NumPy, Pandas

Visualization: Matplotlib, Seaborn

Environment: Google Colab / Jupyter Notebook / VS Code
