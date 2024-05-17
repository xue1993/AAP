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
W = zeros(size(U, 1), k);
W(:, 1) = U(:, 1);

for j = 2:k
    C = U(:, j) * V(:, j)';
    C = C .* (C >= 0);    
    [u, s, v] = svds(C, 1);
    W(:, j) = u;
end

% Display the resulting W matrix
disp(W);

% Define a convergence criterion
max_iters = 100;
tolerance = 1e-10;
prev_error = Inf;
converged = false;
iter = 0;

while ~converged && iter < max_iters
    iter = iter + 1;
    
    % Normalize W
    W = W ./ sqrt(sum(W.^2, 1));
    
    % Update H
    for i = 1:n
        H(:, i) = lsqnonneg(W, A(:, i));
    end
    
    % Update W
    for i = 1:m
        W(i, :) = lsqnonneg(H', A(i, :)')';
    end
    
    % Calculate approximation error
    approx_A = W * H;
    error = norm(A - approx_A, 'fro')/norm(A, 'fro');
    display(error)
    
    % Check convergence
    if abs(prev_error - error) < tolerance
        converged = true;
    end
    prev_error = error;
end

% Display the resulting W and H matrices
disp(iter)
disp('W = ');
disp(W);
disp('H = ');
disp(H);