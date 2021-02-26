close all;  clc;
%load dataset 
data = readmatrix('credit_card_new.csv');

%defining the predictors and labels
x=(data(:,1:16));
y=data(:,17);
m=length(y);

P=0.7; %70 percent of data for training set and 30% for the test set 
idx=randperm(m);
xtrain = x(idx(1:round(P*m)),:);
ytrain=y(idx(1:round(P*m)),:);
xtest = x(idx(round(P*m)+1:end),:);
ytest=y(idx(round(P*m)+1:end),:);

%Using for-loops to find the best  values for the hyperparameters of
%DistributionNames and Priors
%{
DistributionName = {'Normal','kernel',};
prior={'empirical','uniform'};
for i=1:length(DistributionName)
    for j=1:length(prior)
        rng default
        Mdl = fitcnb(xtrain,ytrain,'ClassNames',{'0','1'},'Prior',char(prior(j)),...
                     'DistributionNames',char(DistributionName(i)));       
        losses(i,j)=resubLoss(Mdl);   
        
    end
end  
losses; 
%}
%result-losses for prior ='empirical' and Distribution ='Normal' was the lowest hence using it
%into in the model 

tic;
rng('default') 
Mdl = fitcnb(xtrain,ytrain,'ClassNames',{'0','1'},'Prior','empirical','DistributionNames','normal');
toc;

%predicting on training set 
%{
[label_train, score_train] =predict(Mdl,xtrain);
ypredtrain = str2double(label_train);

%accuracy,Precision,Recall,F1_score and plotting ROC Curves on test data
figure
cm = confusionchart(ytrain,ypredtrain);%confusion chart
accuracy = mean(double(ypredtrain==ytrain))*100;%Accuracy of model

confMat=confusionmat(ytrain,ypredtrain);
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(i,:)); 
end
precision(isnan(precision))=[];
Precision=(sum(precision)/size(confMat,1))*100;%Precision of the model

for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(:,i));  
end
Recall=(sum(recall)/size(confMat,1))*100;%Recall of the model

F1_score=2*Recall*Precision/(Precision+Recall);%F1_Score of the model
figure            %code for the ROC curves 
AUC = zeros(1,2);
for class = 1:2
    [x1,y1,~,AUC(class)] = perfcurve(ytrain,score_train(:,class),Mdl.ClassNames(class));
    plot(x1,y1)
    hold on
    legend('defaulter','non-defaulter')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curves');
end
%}

%predicting on test data 
[label_test, score_test] =predict(Mdl,xtest);
ypredtest = str2double(label_test);

%accuracy,Precision,Recall,F1_score and plotting ROC Curves on test data
figure
cm = confusionchart(ytest,ypredtest);%confusion chart
accuracy = mean(double(ypredtest==ytest))*100;%Accuracy of model

confMat=confusionmat(ytest,ypredtest);%confusion matrix
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(i,:)); 
end
precision(isnan(precision))=[];
Precision=(sum(precision)/size(confMat,1))*100;%Precision of the model

for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(:,i));  
end
Recall=(sum(recall)/size(confMat,1))*100;%Recall of the model

F1_score=2*Recall*Precision/(Precision+Recall);%F1_Score of the model
figure    %code for the ROC curves
AUC = zeros(1,2);
for class = 1:2
    [x1,y1,~,AUC(class)] = perfcurve(ytest,score_test(:,class),Mdl.ClassNames(class));
    plot(x1,y1)
    hold on
    legend('defaulter','non-defaulter')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curves');
end

