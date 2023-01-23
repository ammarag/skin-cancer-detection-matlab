clc
clear all
close all

load('svm_reduced_features.mat');

Data_Class=[];
Data_Class=svm_reduced_features; 
Diag=[ones(58,1);zeros(18,1)];

groups = Diag;

k=2;
c = cvpartition(76,'kfold',k);
for i = 1:c.NumTestSets
trIdx = c.training(i);
teIdx = c.test(i);
end
TrainSet=find(trIdx==1);
TestSet=find(teIdx==1);
Diag_T=Diag(TrainSet);
Diag_Te=Diag(TestSet);
Corr_1=Data_Class(:,1);
Corr_2=Data_Class(:,2);
Corr_3=Data_Class(:,3);
Corr_4=Data_Class(:,4);
Corr_5=Data_Class(:,5);

Train_Data=[Corr_1(TrainSet)  Corr_2(TrainSet) Corr_3(TrainSet) Corr_4(TrainSet) Corr_5(TrainSet)];
Test_Data=[Corr_1(TestSet)  Corr_2(TestSet) Corr_3(TestSet)  Corr_4(TestSet)  Corr_5(TestSet)  ];

z=[1,1];
minfn = @(z)crossval('mcr',Data_Class,Diag,'Predfun', ...
    @(xtrain,ytrain,xtest)crossfun(xtrain,ytrain,...
    xtest,exp(z(1)),exp(z(2))),'partition',c);
opts = optimset('TolX',5e-4,'TolFun',5e-4);
[searchmin, fval]=fminsearch(minfn,randn(2,1),opts);
z=exp(searchmin);

% figure
svmStructN = svmtrain(Train_Data,Diag_T,'Kernel_Function','rbf',...
'rbf_sigma',z(1),'boxconstraint',z(2),'showplot',true);


 Results = svmclassify(svmStructN,Test_Data,'showplot',true);
 

 
 cp = classperf(groups);
 classperf(cp,Results,teIdx);
cp.CorrectRate


