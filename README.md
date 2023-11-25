# Supervised-Image-Classification
This code was developed as a project for the Supervised Learning Exam of the Master's Degree in Artificial Intelligence for Science and Technology.

The goal of this project is to perform image classification on a given dataset, which is a variant of the Tiny ImageNet dataset. The task is solved as a supervised learning problem, with two different approaches.
As first method it is used a model based on SIFT and Bag of Words (BoW), followed by a Support Vector Classifier (SVC) as a traditional classifier.
The second approach involved using a model based on a Convolutional Neural Network (CNN); to obtain this it was used the REGNET_X_1_6GF pre-trained PyTorch model, of which the final classifier was changed and then trained with transfer learning.

The first model of the two was implemented both in Matlab (Supervised_Final_Exam_SIFT_BOW_Canesi_Casati.m) and Python (Supervised_Final_Exam_SIFT_BOW_Canesi_Casati.ipynb), while the second one only in Python (Supervised_Final_Exam_CNN_Canesi_Casati.ipynb).
The repository also contains the Final Report of the project (Exam_Project_Supervised_CANESI_CASATI.pdf), with a final evaluation on the performances of the algorithms and a comparison between them.
