function nn
clear;close all;
addpath('nn')
load('dataset')
X = dataset(:,1:2);
y = dataset(:,3);
data_mu = 0;
data_s=1;

nn_lambda = .001;
nn_lsizes = [2 20 1];
nn_options = optimset('MaxIter', 600);
  
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, y, nn_lambda), nnInitParams(nn_lsizes), nn_options);
    
accuracy = mean(  (  y==round(nnFeedForward(nn_params, nn_lsizes, X))  )(:)  );
printf(" %d", nn_lsizes), printf("\t| %10.2f%%\n", accuracy*100);

%% Testing
test_in = nn_input = ([1 1]);
test_out = nnFeedForward(nn_params, nn_lsizes, nn_input);
[test_in test_out]

hold on;
scatter3(X(find(y==0),1), X(find(y==0),2), 0*X(find(y==0),2), 'r');
scatter3(X(find(y==1),1), X(find(y==1),2), 0*X(find(y==1),2), 'g');
mesh(linspace(min(X(:,1)),max(X(:,1)), 2), linspace(min(X(:,2)),max(X(:,2)), 2), zeros(2, 2));

input1_linspace = linspace(min(X(:,1)), max(X(:,1)), 10);
input2_linspace = linspace(min(X(:,2)), max(X(:,2)), 10);
[input1_matrix, input2_matrix] = meshgrid(input1_linspace, input2_linspace);
output_matrix = [];
for i = 1:length(input1_linspace)
    for j = 1:length(input2_linspace)
        nn_input = ([input1_linspace(i) input2_linspace(j)]-data_mu)./data_s;
        output_matrix(i,j) = nnFeedForward(nn_params, nn_lsizes, nn_input) -0.5;
    end
end
mesh(input1_linspace, input2_linspace, output_matrix);

endfunction

