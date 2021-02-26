close all; clear all; clc;
%load dataset 
data = readmatrix('credit_card_new.csv');

%defining the dependent and the independent variables 
x=(data(:,1:16));
y=data(:,17);
m=length(y);

P=0.7; %70 percent of data for training set and 30% for the test set 
idx=randperm(m);
xtrain = x(idx(1:round(P*m)),:);
ytrain=y(idx(1:round(P*m)),:);
xtest = x(idx(round(P*m)+1:end),:);
ytest=y(idx(round(P*m)+1:end),:);
rng('default');

%model using lasso regularisation to find the significant features 

tic;
rng('default')
lambda(1)=1.0;
for i=2:50
    lambda(i)=lambda(i-1)*0.75;%iterating through decreasing values of lambda, it decreases by 0.75 in this case
end 
[B,FitInfo] = lassoglm(xtrain,ytrain,'binomial','CV',10,'Lambda',lambda);
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show','Location','best');
indx = FitInfo.Index1SE;% reference:https://in.mathworks.com/help/stats/regularize-logistic-regression.html
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0);
cnst = FitInfo.Intercept(indx);
B1 = [cnst;B0];
predictors = find(B0); %  finding the indices of nonzero predictors
rng('default')
Mdl = fitglm(xtrain,ytrain,'linear','Distribution','binomial','PredictorVars',predictors);% Model
toc;

%predicting on the train set 
%{
[ypredtrain, score_train]=predict(Mdl,xtrain);
ypredtrain(ypredtrain>=0.5)=1;
ypredtrain(ypredtrain<0.5)=0;

accuracy = mean(double(ypredtrain==ytrain))*100;%Accuracy of model
figure
cchart = confusionchart(ytrain,ypredtrain);%confusion chart

confMat=confusionmat(ytrain,ypredtrain);%confusion matrix
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(i,:)); 
end
precision(isnan(precision))=[];
Precision=(sum(precision)/size(confMat,1))*100;%Precision of the model

for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(:,i));  
end

Recall=(sum(recall)/size(confMat,1))*100;%Recall of the model

F1_score=(2*Recall*Precision/(Precision+Recall));%F1_Score of the model
    
figure   %code for the ROC curves 
AUC = zeros(1,2);
z=[0,1];
for class = 1:2
    [x1,y1,~,AUC(class)] = perfcurve(ytrain,score_train(:,class),z(class));
    plot(x1,y1);
    hold on
    legend('defaulter','non-defaulter')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curves');
end
%}

%predicting on the test set 
[ypredtest, score_test]=predict(Mdl,xtest);
ypredtest(ypredtest>=0.5)=1;
ypredtest(ypredtest<0.5)=0;


accuracy = mean(double(ypredtest==ytest))*100;%Accuracy of model
figure
cchart = confusionchart(ytest,ypredtest);%confusion chart

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

F1_score=(2*Recall*Precision/(Precision+Recall));%F1_Score of the model

figure                               %code for the ROC curves 
AUC = zeros(1,2);
z=[0,1];
for class = 1:2
    [x1,y1,~,AUC(class)] = perfcurve(ytest,score_test(:,class),z(class));
    plot(x1,y1);
    hold on
    legend('defaulter','non-defaulter')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curves');
end

