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

% Initialize W
W = rand(m, k);
H = zeros(k, n);


svd_init = true; % Correct boolean value in MATLAB

if svd_init
    % Perform singular value decomposition (SVD)
    [U, S, V] = svds(A, k);
    W(:, 1) = U(:, 1);
    for j = 2:k
        C = U(:, j) * V(:, j)';
        C = C .* (C >= 0);    
        [u, s, v] = svds(C, 1);
        W(:, j) = u;
    end
    W = abs(W);
end

% Display the resulting W matrix
disp(W);

% Define a convergence criterion
max_iters = 100;
converged = false;
iter = 0;

ERROR_TRUTH = 0.4095009840988514; %0.4095594010470482;
errors = []; % Array to store error values

error = norm(A - W * H, 'fro') / norm(A, 'fro');
errors = [errors; error- ERROR_TRUTH];
fprintf('Iteration %d, Error: %.16f\n', iter, error);

m_AA = 2;
WH_list = [];
residuals_list = [];

while iter < max_iters
    WH_flattened = [W(:); H(:)];
    WH_list = [WH_list, WH_flattened];

    iter = iter + 1;
    
    % Normalize W
    W_ = W ./ sqrt(sum(W.^2, 1));
    
    % Update H
    for i = 1:n
        H_(:, i) = lsqnonneg(W_, A(:, i));
    end
    
    % Update W
    for i = 1:m
        W_(i, :) = lsqnonneg(H_', A(i, :)')';
    end

    WH_flattened = [W_(:); H_(:)];
    
    % Compute residuals and append to residuals_list
    residuals = WH_flattened - WH_list(:, end);
    residuals_list = [residuals_list, residuals];

    if size(WH_list, 2) > m_AA+1
        WH_list = WH_list(:, 2:end);
        residuals_list = residuals_list(:,2:end);
    end

    % Compute gVec, S, Y
    gVec = residuals_list(:, end);
    S = diff(WH_list, 1, 2);
    Y = diff(residuals_list, 1, 2);
    
    fprintf('S.shape: [%d, %d]\n', size(S, 1), size(S, 2));
    
    % Solve the unconstrained LS problem
    alpha_lstsq = linsolve(Y, gVec);

    
    % Update AAP direction with alpha_lstsq solution
    damping = 1; % Define your damping factor
    pVec = -damping * gVec +  (S + damping * Y) * alpha_lstsq;
    
    fprintf('Length of ptilde: %.16f, rtilde: %.16f\n', norm(S * alpha_lstsq), norm(Y * alpha_lstsq - gVec));
    
    % Update W and H
    W = W - reshape(pVec(1:numel(W)), size(W));
    H = H - reshape(pVec(numel(W)+1:end), size(H));
        
    % Calculate approximation error
    approx_A = W * H;
    error = norm(A - approx_A, 'fro') / norm(A, 'fro'); 
    errors = [errors; error- ERROR_TRUTH]; % Store the error
    fprintf('Iteration %d, Error: %.16f\n', iter, error);
        
        
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