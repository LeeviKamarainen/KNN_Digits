function C = knn_pa(trainclass, traindata, data,k)
% function C = knn_pa(trainclass, traindata, data,k)
% The function KNN_PA performs calculations related to Euclidean distances
% between training and testing points and classifications of the testing
% data with kNN-algorithm. 
%
% INPUT  trainclass     row vector containing the classes of the examples,
%                       so that element i of trainclass is the class of the
%                       example in column i of traindata.
%       traindata       matrix containing training examples so that each
%                       column is a single stroke, and all of its datapoints. 
%       data            matrix containing samples to be classified, one
%                       stroke in each column, with all of its datapoints. 
%
% OUTPUT C              row vector of classes and it should include one
%                       value for each column in data 


%% Calculate Euclidean distances between (test) data and training points

% Utilised the code used in Exercise 9 and modified it.

%We will calculate the nearest distance for each of the datapoints in the
%sample to be classified compared to datapoints in the training samples and vice versa. 
%We will then take the mean value of the distances,
%meaning that as a result we will get the average distance for each point
%in sample we want to classify to the training samples and vice versa.

%Init distance and class matrix
traindistances = []; %Dimensions will be = Amount of samples to classify x Amount of training samples
classmat = []; %A class matrix so each of the distances will have the corresponding classes
for sample = 1:1:length(data) %All of the samples that we want to classify;
    for trsample = 1:1:length(traindata)
        trainpoints = traindata(trsample).pos(:,1:2); %All of the x- and y-values from a single training sample
        datapoint = data(sample).pos(:,1:2); %Only the x- and y-values to the datapoint
        
        [nearestInd nearestDist] = dsearchn(datapoint,trainpoints); %Find the nearest points of sample to be classified from the training sample
        [nearestInd2 nearestDist2] = dsearchn(trainpoints,datapoint); %Find the nearest points of training sample from the sample to be classified
        traindistances(sample,trsample) = mean(nearestDist.^2)+mean(nearestDist2.^2); %We raise the distances to power of 2 so larger distances get "punished".
    end
    classmat(sample,:) = trainclass; %Create corresponding class matrix (every distance has the same class as it was defined on trainclass)
end
%% Sort the distances to increasing order

[sorteddist, I] = sort(traindistances,2); %Sort the traindistances matrix row wise
%% Sort the class information to correspond with the sorted data points
classsorted = [];
for i = 1:1:length(data)
    classsorted(i,:) = classmat(i,I(i,:));
end
%% Select the k nearest data points and their classes
nearestclass = [];
nearest = [];
for i = 1:1:length(data)
   nearest(i,:) = sorteddist(i,1:k); 
   nearestclass(i,:) = classsorted(i,1:k);
end
%% Perform the classification of the data

% Select new class as the one which is the most common one in k-nearest
% neighbors
newdataclass = [];
for i = 1:1:size(data,2);
    newdataclass(i) = mode(nearestclass(i,:)); %% Mode returns most common value in array
end


% Determine the class labels
C = newdataclass;
end