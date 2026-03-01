# Supervised Image Classification

This repository contains a project developed for the Supervised Learning exam of the Master's Degree in Artificial Intelligence for Science and Technology.

The goal of the project is to perform image classification on a dataset derived from Tiny ImageNet. The task is approached as a supervised learning problem using two different methods:

1. A traditional computer vision pipeline based on SIFT and Bag of Words (BoW), followed by a Support Vector Classifier (SVC).
2. A deep learning approach based on a Convolutional Neural Network (CNN), using the pre-trained PyTorch model `REGNET_X_1_6GF`. The final classifier was replaced and then trained through transfer learning.

The SIFT + BoW model was implemented in both MATLAB and Python:

- `Supervised_Final_Exam_SIFT_BOW_Canesi_Casati.m`
- `Supervised_Final_Exam_SIFT_BOW_Canesi_Casati.ipynb`

The CNN-based model was implemented in Python:

- `Supervised_Final_Exam_CNN_Canesi_Casati.ipynb`

The repository also includes the final project report:

- `Exam_Project_Supervised_CANESI_CASATI.pdf`

The report contains the final evaluation of the algorithms' performance and a comparison between the two approaches.
