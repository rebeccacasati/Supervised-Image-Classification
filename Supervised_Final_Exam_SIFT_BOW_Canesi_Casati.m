%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%               Exam Project Supervised Learning - June 2023 -            %
%                   Canesi Gabriele, Casati Rebecca                       %
%                                                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Variables, Close Current Figures, and Create Results Directory 
clc;
clear all;
close all;
%% Load Variable

load bag_final.mat;

%% Configurations 

classes = 0:99; % Should be sorted alphapetically to match test data automatic labeling by folder
numberOfClusters = 500; % The number of clusters representing the number of features in the bag of features. Default can be 500.
ratioOfStrongFeatures = 0.85; % Default can be 0.8
SVM_Kernel = 'rbf'; % Can be either 'polynomial' or 'rbf' for example.
SVM_C = 0.1; % Smaller is less overfitting. Default can be 0.1.
SVM_RBF_Gamma = 0.6; % The RBF SVM Kernel gamma. The higher the more complex model, and more prune to overfitting. Default can be 0.6.

%% Data stores images for the train, validation and test set

imgDS_train_val = imageDatastore(fullfile("TinyImageNet","train"), "IncludeSubfolders",true,"LabelSource","foldernames") ;
imgDS_test = imageDatastore(fullfile("TinyImageNet","val"), "IncludeSubfolders",true,"LabelSource","foldernames") ;

%% Partition Training and Validation (80% training 20% validation)

[TrainSet, ValSet] = splitEachLabel(imgDS_train_val, 0.8, "randomized");

%% Extracting features with Surf and create visual words with k-means

bag = bagOfFeatures(TrainSet,"StrongestFeatures" ,ratioOfStrongFeatures,'VocabularySize', numberOfClusters);

%% Visualize and save one instogram

visualize_sample_FV = 1;
if (visualize_sample_FV)
    figure
    img = readimage(imgDS_train_val,10); % First image of first class as an example
    featureVector = encode(bag, img);
    % Plot the histogram of visual word occurrences
    figure;
    bar(featureVector);
    title('Visual word occurrences');
    xlabel('Visual word index');
    ylabel('Frequency of occurrence');
    saveas(gcf,'Results//TrainData-SampleFeatureVector.png');
end

%% Defining and train the SVC Classifier

opts = templateSVM('KernelFunction', SVM_Kernel ,'BoxConstraint', SVM_C, 'kernelScale', SVM_RBF_Gamma);
classifier = trainImageCategoryClassifier(TrainSet, bag, 'LearnerOptions', opts);

%% Evaluate the classifier on training then on validation data

confMatrix_train = evaluate(classifier, TrainSet);
confMatrix_val = evaluate(classifier, ValSet);
tran_val_avg_accuracy = (mean(diag(confMatrix_val)) + mean(diag(confMatrix_train))) / 2; % This information should be used to tweak the system parameters for better accuracy
display(['The training and validation average accuracy is ' num2str(tran_val_avg_accuracy) '.']);

%% Evaluation of the model on the test set

confMatrix_test = evaluate(classifier, imgDS_test);
test_accuracy = mean(diag(confMatrix_test));
display(['The test accuracy is ' num2str(test_accuracy) '.']);

%% Save the bag of words parameters 
save("bag_final.mat", "bag")