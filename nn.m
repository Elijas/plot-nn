function nn
clear;close all;
addpath('nn')
load('dataset')

%%%%[ NN Training ]%%%%
[X, y, data_mu, data_s] = prepareData(dataset);
if max(y)>1, yExp = eye(max(y))(y,:); else, yExp=y; end % For a multi-class classification problem, expand y to a binary matrix (to prepare for NN learning with multiple units in the output layer)

nn_lambda = .5;
nn_lsizes = [size(X,2)  2   max(y)];
nn_options = optimset('MaxIter', 1000);
printf("# Experiment initializing..\n")
printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
nn_lsizes = nn_lsizes(find(nn_lsizes)); % Eliminates zeros in the vector
    
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, yExp, nn_lambda), nnInitParams(nn_lsizes), nn_options);
    
accuracy = mean(  (  yExp==round(nnFeedForward(nn_params, nn_lsizes, X))  )(:)  );
printf(" %d", nn_lsizes), printf("\t| %10.2f%%\n", accuracy*100);

%%%%[ NN Testing ]%%%%
%One value
nn_input = ([1 1]-data_mu)./data_s;
nnFeedForward(nn_params, nn_lsizes, nn_input);

%Plot training examples
hold on;
plot(X(find(y==0),1), X(find(y==0),2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(X(find(y==1),1), X(find(y==1),2), 'ko', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

%Plot NN outputs
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
mesh(linspace(min(X(:,1)),max(X(:,1)), 2), linspace(min(X(:,2)),max(X(:,2)), 2), zeros(2, 2));

endfunction

