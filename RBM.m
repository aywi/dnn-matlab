% Restricted Boltzmann Machines
classdef RBM
    properties
        sizes
        g
        gr
        x_start
        x_sample
        h_x_sample
        W
        b
        c
        batch_size
    end
    
    properties (Access = protected)
        x_neg
        nabla_W
        nabla_b
        nabla_c
        update_W
        update_b
        update_c
    end
    
    methods
        function rbm = RBM(sizes, g_type)
            rbm.sizes = sizes;
            switch g_type
                case 'bb'
                    a = 1;
                    rbm.g = @(x) 1 ./ (1 + exp(-x));
                    rbm.gr = @(x) rbm.g(x);
                    rbm.x_start = @(x) randi([0, 1], size(x), 'single');
                    rbm.x_sample = @(x) single(rand(size(x)) < x);
                    rbm.h_x_sample = @(x) rbm.x_sample(x);
                case 'gb'
                    a = 1;
                    rbm.g = @(x) 1 ./ (1 + exp(-x));
                    rbm.gr = @(x) x;
                    rbm.x_start = @(x) randn(size(x), 'single');
                    rbm.x_sample = @(x) randn(size(x), 'single') + x;
                    rbm.h_x_sample = @(x) single(rand(size(x)) < x);
                case 'rr'
                    a = 2;
                    rbm.g = @(x) max(x, zeros(size(x), 'single'));
                    rbm.gr = @(x) rbm.g(x);
                    rbm.x_start = @(x) max(randn(size(x), 'single'), zeros(size(x), 'single'));
                    rbm.x_sample = @(x) max(randn(size(x), 'single') + x, zeros(size(x), 'single'));
                    rbm.h_x_sample = @(x) rbm.x_sample(x);
                case 'gr'
                    a = 2;
                    rbm.g = @(x) max(x, zeros(size(x), 'single'));
                    rbm.gr = @(x) x;
                    rbm.x_start = @(x) randn(size(x), 'single');
                    rbm.x_sample = @(x) randn(size(x), 'single') + x;
                    rbm.h_x_sample = @(x) max(randn(size(x), 'single') + x, zeros(size(x), 'single'));
            end
            rbm.W = cell(1, size(rbm.sizes, 2) - 1);
            rbm.b = cell(1, size(rbm.sizes, 2) - 1);
            rbm.c = cell(1, size(rbm.sizes, 2) - 1);
            rbm.x_neg = cell(1, size(rbm.sizes, 2) - 1);
            rbm.nabla_W = cell(1, size(rbm.sizes, 2) - 1);
            rbm.nabla_b = cell(1, size(rbm.sizes, 2) - 1);
            rbm.nabla_c = cell(1, size(rbm.sizes, 2) - 1);
            rbm.update_W = cell(1, size(rbm.sizes, 2) - 1);
            rbm.update_b = cell(1, size(rbm.sizes, 2) - 1);
            rbm.update_c = cell(1, size(rbm.sizes, 2) - 1);
            rng('default');
            for i = 1:size(rbm.sizes, 2) - 1
                rbm.W{i} = randn(rbm.sizes(i + 1), rbm.sizes(i), 'single') .* sqrt(a / rbm.sizes(i));
                rbm.b{i} = zeros(rbm.sizes(i + 1), 1, 'single');
                rbm.c{i} = zeros(rbm.sizes(i), 1, 'single');
                rbm.nabla_W{i} = zeros(rbm.sizes(i + 1), rbm.sizes(i), 'single');
                rbm.nabla_b{i} = zeros(rbm.sizes(i + 1), 1, 'single');
                rbm.nabla_c{i} = zeros(rbm.sizes(i), 1, 'single');
                rbm.update_W{i} = zeros(rbm.sizes(i + 1), rbm.sizes(i), 'single');
                rbm.update_b{i} = zeros(rbm.sizes(i + 1), 1, 'single');
                rbm.update_c{i} = zeros(rbm.sizes(i), 1, 'single');
            end
            rbm.batch_size = 1;
        end
        
        function [rbm, reconstruction_errors] = train(rbm, data_train, data_valid, batch_size, epoch, alpha, l1, l2, momentum, k, k_pcd, use_gpu, output)
            x_train = single(data_train{1});
            n_train = size(x_train, 2);
            x_valid = single(data_valid{1});
            rbm.batch_size = batch_size;
            train_reconstruction_errors = zeros(epoch, 1, 'single');
            valid_reconstruction_errors = zeros(epoch, 1, 'single');
            if use_gpu
                x_train = gpuArray(x_train);
                x_valid = gpuArray(x_valid);
                train_reconstruction_errors = gpuArray(train_reconstruction_errors);
                valid_reconstruction_errors = gpuArray(valid_reconstruction_errors);
                for i = 1:size(rbm.sizes, 2) - 1
                    rbm.W{i} = gpuArray(rbm.W{i});
                    rbm.b{i} = gpuArray(rbm.b{i});
                    rbm.c{i} = gpuArray(rbm.c{i});
                    rbm.nabla_W{i} = gpuArray(rbm.nabla_W{i});
                    rbm.nabla_b{i} = gpuArray(rbm.nabla_b{i});
                    rbm.nabla_c{i} = gpuArray(rbm.nabla_c{i});
                    rbm.update_W{i} = gpuArray(rbm.update_W{i});
                    rbm.update_b{i} = gpuArray(rbm.update_b{i});
                    rbm.update_c{i} = gpuArray(rbm.update_c{i});
                end
            end
            rng('default');
            for i = 1:epoch
                shuffle = randperm(n_train);
                x_train = x_train(:, shuffle);
                for j = 1:rbm.batch_size:n_train
                    x_batch = x_train(:, j:min(j + rbm.batch_size - 1, end));
                    rbm = rbm.SGD(x_batch, alpha, l1, l2, momentum, k, k_pcd);
                end
                [~, ~, train_reconstruction_errors(i)] = rbm.reconstruct({x_train}, true, false);
                [~, ~, valid_reconstruction_errors(i)] = rbm.reconstruct({x_valid}, true, false);
                if output
                    fprintf('Epoch %d completed\n', i);
                    fprintf('Training reconstruction error  : %7.4f\n', train_reconstruction_errors(i));
                    fprintf('Validation reconstruction error: %7.4f\n', valid_reconstruction_errors(i));
                end
            end
            for i = 1:size(rbm.sizes, 2) - 1
                rbm.x_neg{i} = [];
                rbm.nabla_W{i} = zeros(rbm.sizes(i + 1), rbm.sizes(i), 'single');
                rbm.nabla_b{i} = zeros(rbm.sizes(i + 1), 1, 'single');
                rbm.nabla_c{i} = zeros(rbm.sizes(i), 1, 'single');
                rbm.update_W{i} = zeros(rbm.sizes(i + 1), rbm.sizes(i), 'single');
                rbm.update_b{i} = zeros(rbm.sizes(i + 1), 1, 'single');
                rbm.update_c{i} = zeros(rbm.sizes(i), 1, 'single');
            end
            if use_gpu
                train_reconstruction_errors = gather(train_reconstruction_errors);
                valid_reconstruction_errors = gather(valid_reconstruction_errors);
                for i = 1:size(rbm.sizes, 2) - 1
                    rbm.W{i} = gather(rbm.W{i});
                    rbm.b{i} = gather(rbm.b{i});
                    rbm.c{i} = gather(rbm.c{i});
                end
            end
            reconstruction_errors = {train_reconstruction_errors, valid_reconstruction_errors};
        end
        
        function [x_test_encode, x_test_reconstruct, test_reconstruction_error] = reconstruct(rbm, data_test, test_error, output)
            x_test = single(data_test{1});
            n_test = size(x_test, 2);
            h_x = cell(1, size(rbm.sizes, 2));
            h_x{1} = x_test;
            for i = 2:size(rbm.sizes, 2)
                h_x{i} = rbm.g(rbm.W{i - 1} * h_x{i - 1} + repmat(rbm.b{i - 1}, 1, n_test));
            end
            x_test_encode = h_x(end);
            h_x_reconstruct = cell(1, size(rbm.sizes, 2));
            h_x_reconstruct{end} = h_x{end};
            for i = size(rbm.sizes, 2) - 1:-1:1
                h_x_reconstruct{i} = rbm.gr(rbm.W{i}' * h_x_reconstruct{i + 1} + repmat(rbm.c{i}, 1, n_test));
            end
            x_test_reconstruct = h_x_reconstruct{1};
            if test_error
                test_reconstruction_error = rbm.reconstruction_error(x_test, x_test_reconstruct);
                if output
                    fprintf('Test reconstruction error      : %7.4f\n', test_reconstruction_error);
                end
            end
        end
    end
    
    methods (Access = protected)
        function rbm = SGD(rbm, x, alpha, l1, l2, momentum, k, k_pcd)
            n = size(x, 2);
            [rbm, h_x, h_x_neg] = rbm.CD_k(x, k, k_pcd);
            for i = 1:size(rbm.sizes, 2) - 1
                rbm.nabla_W{i} = (h_x_neg{i + 1} * h_x_neg{i}') ./ k_pcd - (h_x{i + 1} * h_x{i}') ./ n;
                rbm.nabla_b{i} = sum(h_x_neg{i + 1}, 2) ./ k_pcd - sum(h_x{i + 1}, 2) ./ n;
                rbm.nabla_c{i} = sum(h_x_neg{i}, 2) ./ k_pcd - sum(h_x{i}, 2) ./ n;
                rbm.update_W{i} = momentum .* rbm.update_W{i} - alpha .* (rbm.nabla_W{i} + l1 .* sign(rbm.W{i}) + l2 .* rbm.W{i});
                rbm.update_b{i} = momentum .* rbm.update_b{i} - alpha .* (rbm.nabla_b{i});
                rbm.update_c{i} = momentum .* rbm.update_c{i} - alpha .* (rbm.nabla_c{i});
                rbm.W{i} = rbm.W{i} + rbm.update_W{i};
                rbm.b{i} = rbm.b{i} + rbm.update_b{i};
                rbm.c{i} = rbm.c{i} + rbm.update_c{i};
            end
        end
        
        function [rbm, h_x, h_x_neg] = CD_k(rbm, x, k, k_pcd)
            n = size(x, 2);
            h_x = cell(1, size(rbm.sizes, 2));
            h_x{1} = x;
            h_x_neg = cell(1, size(rbm.sizes, 2));
            rng('default');
            for i = 2:size(rbm.sizes, 2)
                h_x{i} = rbm.g(rbm.W{i - 1} * h_x{i - 1} + repmat(rbm.b{i - 1}, 1, n));
                if isempty(rbm.x_neg{i - 1})
                    rbm.x_neg{i - 1} = rbm.x_start(zeros(size(h_x{i - 1}, 1), k_pcd));
                end
                h_x_neg{i} = rbm.g(rbm.W{i - 1} * rbm.x_neg{i - 1} + repmat(rbm.b{i - 1}, 1, k_pcd));
                for j = 1:k
                    h_x_sample = rbm.h_x_sample(h_x_neg{i});
                    h_x_neg{i - 1} = rbm.gr(rbm.W{i - 1}' * h_x_sample + repmat(rbm.c{i - 1}, 1, k_pcd));
                    x_sample = rbm.x_sample(h_x_neg{i - 1});
                    h_x_neg{i} = rbm.g(rbm.W{i - 1} * x_sample + repmat(rbm.b{i - 1}, 1, k_pcd));
                end
                rbm.x_neg{i - 1} = h_x_neg{i - 1};
            end
        end
    end
    
    methods (Access = protected, Static)
        function error = reconstruction_error(x, x_reconstruct)
            error = sum(mean((x_reconstruct - x) .^ 2, 1), 2) / size(x, 2);
        end
    end
end
