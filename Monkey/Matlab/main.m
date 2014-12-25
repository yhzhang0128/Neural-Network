clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('full234.mat');
m = size(X, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Initialize Theta Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

%load('init.mat')
%fprintf('init: %d\n', size(initial_nn_params));
%initial_Theta1 = reshape(initial_nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 hidden_layer_size, (input_layer_size + 1));

%initial_Theta2 = reshape(initial_nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 num_labels, (hidden_layer_size + 1));

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Start Training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Training Neural Network... \n')

options = optimset('MaxIter', 200);

% lambda is for Normalization
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)

% Optimization
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save Theta.mat Theta1 Theta2
fprintf('Program Finished. Press enter to continue.\n');
pause;
