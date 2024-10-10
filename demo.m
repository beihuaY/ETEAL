warning off;
clc;
clear all;


datapath = './';
dataname = 'NH_p4660';
load([dataname,'.mat']); 
fprintf('Dataset:%s\t \n',dataname);
addpath(genpath('./'));

%%
nv=length(X); % view num
num=length(gt); % sample num
ns=length(unique(gt)); 

%% Parameter Setting 
      dim_H = 200;
      param.alpha = 2000;
      param.lambda =  1000;
     
      param.fun = 'laplace';
      param.gamma = 0.1;
      anchor = 1;
      numanchor = anchor*ns;

for iv=1:nv 
    X{iv}=NormalizeFea(X{iv},0);
end
     
labels = ETEAL(X,gt,numanchor,param,dim_H);
result = Clustering8Measure(labels,gt) ; % 'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'
fprintf('acc:%5.4f\tnmi:%5.4f\tpur:%5.4f\tFs:%5.4f\t\n',[result(1) result(2) result(3) result(4)]);





     
