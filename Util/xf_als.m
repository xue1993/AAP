clear all
% Given matrix A and number of singular values k
A = [
    0.0, 0, 0, 1, 0;
    0, 0, 0, 0, 1;
    0, 0, 0, 0, 1;
    1, 0, 1, 0, 0;
    1, 0, 0, 0, 0;
    0, 1, 0, 0, 0;
    1, 0, 1, 1, 0;
    0, 1, 1, 0, 0;
    0, 0, 1, 1, 1;
    0, 1, 1, 0, 0
];
m = 10;
n = 5;
k = 3;

% Perform singular value decomposition (SVD)
[U, S, V] = svds(A, k);

% Initialize W
W = zeros(m, k);
H = zeros(k, n);
W(:, 1) = U(:, 1);

for j = 2:k
    C = U(:, j) * V(:, j)';
    C = C .* (C >= 0);    
    [u, s, v] = svds(C, 1);
    W(:, j) = u;
end
W = abs(W);

% Display the resulting W matrix
disp(W);

% Define a convergence criterion
max_iters = 100;
prev_error = Inf;
converged = false;
iter = 0;

ERROR_TRUTH = 0.4095009840988514; %0.4095594010470482;
errors = []; % Array to store error values

error = norm(A - W * H, 'fro') / norm(A, 'fro') - ERROR_TRUTH;
errors = [errors; error];

while iter < max_iters
    iter = iter + 1;
    
    % Normalize W
    W = W ./ sqrt(sum(W.^2, 1));
    
    % Update H
    H = linsolve(W, A);
    H(H < 0) = 0;

    % Update W
    W = linsolve(H', A')';
    W(W < 0) = 0;
    
    
    % Calculate approximation error
    approx_A = W * H;
    error = norm(A - approx_A, 'fro') / norm(A, 'fro'); 
    errors = [errors; error- ERROR_TRUTH]; % Store the error
    fprintf('Iteration %d, Error: %.16f\n', iter, error);
    
    
    prev_error = error;
end

% Display the resulting W and H matrices
disp(iter)
disp('W = ');
disp(W);
disp('H = ');
disp(H);

% Plot the error
figure;
semilogy(1:length(errors), errors, '-o');
xlabel('Iteration');
ylabel('Error');
title('Convergence Plot');
grid on;