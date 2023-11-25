# Supervised-Image-Classification
This code was developed as a project for the Supervised Learning Exam of the Master's Degree in Artificial Intelligence for Science and Technology.

The goal of this project is to perform image classification on a given dataset, which is a variant of the Tiny ImageNet dataset. The task is solved as a supervised learning problem, with two different approaches.
As first method it is used a model based on SIFT and Bag of Words (BoW), followed by a Support Vector Classifier (SVC) as a traditional classifier.
The second approach involved using a model based on a Convolutional Neural Network (CNN); to obtain this it was used the REGNET_X_1_6GF pre-trained PyTorch model, of which the final classifier was changed and then trained with transfer learning.
The first model of the two was implemented both in Matlab and Python, while the second one only in Python. For what concerns the first model, although the performances of the Matlab implementation were better (accuracy of 13%), the one that was taken into consideration for the evaluation and the comparison with the CNN based model is the Python implementation.
The SIFT/BoW based model performs with a very low accuracy of 6,9%, while on the contrary the CNN performs better, reaching a total accuracy of 70%. Other considerations for evaluating the models were made, leading to the final conclusion that the CNN-based model is clearly better than the other one, and it can still be improved.
