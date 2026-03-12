# emg-spectrogram-classification
Hand gesture classification from EMG spectrograms using CNNs (InceptionV3, ConvNext, ResNet50)
Machine Learning project for hand gesture recognition from EMG signals using spectrogram-based deep learning models.

This project explores the use of Convolutional Neural Networks (CNNs) for classifying Latin alphabet dactylology gestures using spectrogram representations of EMG signals.

The model leverages transfer learning with ResNet50, InceptionV3 and ConvNext to improve classification performance.

EMG Hand Gesture Classification using CNNs

Machine Learning project for hand gesture recognition from EMG signals using spectrogram-based deep learning models.

This project explores the use of Convolutional Neural Networks (CNNs) for classifying Latin alphabet dactylology gestures using spectrogram representations of EMG signals.


**Project Overview**

Electromyography (EMG) signals capture muscle activity and can be used to detect hand gestures.
In this project, EMG signals are transformed into spectrogram images, allowing the use of computer vision models for classification.

**Pipeline:**

EMG signals → Spectrogram generation → CNN model → Gesture classification

**Technologies Used**
Python
NumPy
Pandas
Matplotlib
TensorFlow / Keras
Scikit-learn
Jupyter Notebook

**Model Architecture**

Spectrogram images are used as input
The convolutional layers extract spatial-frequency features
Fully connected layers perform gesture classification
Dataset Processing

**Steps applied to EMG signals:**

Signal preprocessing
Noise filtering
Spectrogram generation using Short-Time Fourier Transform
Image normalization

**Dataset splitting (train / validation / test)**
