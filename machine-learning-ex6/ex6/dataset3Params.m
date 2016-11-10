function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cand_C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
cand_sigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
pred_result = zeros(size(cand_C, 1), size(cand_sigma, 1));

for i = 1:size(cand_C)
    for j = 1:size(cand_sigma)
        model= svmTrain(X, y, cand_C(i), @(x1, x2) gaussianKernel(x1, x2, cand_sigma(j)));
        predictions = svmPredict(model, Xval);
        pred_result(i, j) = mean(double(predictions ~= yval));
    end
end
[C_idx, sigma_idx] = find(pred_result==min(min(pred_result)));

C = cand_C(C_idx);
sigma = cand_sigma(sigma_idx);

fprintf(' Best parameters : C  %f   sigma  %f \n', C, sigma);


% =========================================================================

end
