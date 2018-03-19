% Autoencoders
classdef AE
    properties
        sizes
        g
        g_prime
        W
        b
        c
        batch_size
        dropout
    end
    
    properties (Access = protected)
        nabla_W
        nabla_b
        nabla_c
        update_W
        update_b
        update_c
    end
    
    methods
        function ae = AE(sizes, g_type)
            ae.sizes = sizes;
            switch g_type
                case 'linear'
                    a = 1;
                    ae.g = @(x) x;
                    ae.g_prime = @(x) ones(size(x), 'single');
                case 'sigmoid'
                    a = 1;
                    ae.g = @(x) 1 ./ (1 + exp(-x));
                    ae.g_prime = @(x) ae.g(x) .* (1 - ae.g(x));
                case 'tanh'
                    a = 1;
                    ae.g = @tanh;
                    ae.g_prime = @(x) 1 - tanh(x) .^ 2;
                case 'relu'
                    a = 2;
                    ae.g = @(x) max(x, zeros(size(x), 'single'));
                    ae.g_prime = @(x) single(x > 0);
            end
            rng('default');
            ae.W = randn(ae.sizes(2), ae.sizes(1), 'single') .* sqrt(a / ae.sizes(1));
            ae.b = zeros(ae.sizes(2), 1, 'single');
            ae.c = zeros(ae.sizes(1), 1, 'single');
            ae.nabla_W = zeros(ae.sizes(2), ae.sizes(1), 'single');
            ae.nabla_b = zeros(ae.sizes(2), 1, 'single');
            ae.nabla_c = zeros(ae.sizes(1), 1, 'single');
            ae.update_W = zeros(ae.sizes(2), ae.sizes(1), 'single');
            ae.update_b = zeros(ae.sizes(2), 1, 'single');
            ae.update_c = zeros(ae.sizes(1), 1, 'single');
            ae.batch_size = 1;
            ae.dropout = 0;
        end
        
        function [ae, reconstruction_errors] = train(ae, data_train, data_valid, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output)
            x_train = single(data_train{1});
            n_train = size(x_train, 2);
            x_valid = single(data_valid{1});
            ae.batch_size = batch_size;
            ae.dropout = dropout;
            train_reconstruction_errors = zeros(epoch, 1, 'single');
            valid_reconstruction_errors = zeros(epoch, 1, 'single');
            if use_gpu
                x_train = gpuArray(x_train);
                x_valid = gpuArray(x_valid);
                train_reconstruction_errors = gpuArray(train_reconstruction_errors);
                valid_reconstruction_errors = gpuArray(valid_reconstruction_errors);
                ae.W = gpuArray(ae.W);
                ae.b = gpuArray(ae.b);
                ae.c = gpuArray(ae.c);
                ae.nabla_W = gpuArray(ae.nabla_W);
                ae.nabla_b = gpuArray(ae.nabla_b);
                ae.nabla_c = gpuArray(ae.nabla_c);
                ae.update_W = gpuArray(ae.update_W);
                ae.update_b = gpuArray(ae.update_b);
                ae.update_c = gpuArray(ae.update_c);
            end
            rng('default');
            for i = 1:epoch
                shuffle = randperm(n_train);
                x_train = x_train(:, shuffle);
                for j = 1:ae.batch_size:n_train
                    x_batch = x_train(:, j:min(j + ae.batch_size - 1, end));
                    ae = ae.SGD(x_batch, alpha, l1, l2, momentum);
                end
                [~, ~, train_reconstruction_errors(i)] = ae.reconstruct({x_train}, true, false);
                [~, ~, valid_reconstruction_errors(i)] = ae.reconstruct({x_valid}, true, false);
                if output
                    fprintf('Epoch %d completed\n', i);
                    fprintf('Training reconstruction error  : %7.4f\n', train_reconstruction_errors(i));
                    fprintf('Validation reconstruction error: %7.4f\n', valid_reconstruction_errors(i));
                end
            end
            ae.nabla_W = zeros(ae.sizes(2), ae.sizes(1), 'single');
            ae.nabla_b = zeros(ae.sizes(2), 1, 'single');
            ae.nabla_c = zeros(ae.sizes(1), 1, 'single');
            ae.update_W = zeros(ae.sizes(2), ae.sizes(1), 'single');
            ae.update_b = zeros(ae.sizes(2), 1, 'single');
            ae.update_c = zeros(ae.sizes(1), 1, 'single');
            if use_gpu
                train_reconstruction_errors = gather(train_reconstruction_errors);
                valid_reconstruction_errors = gather(valid_reconstruction_errors);
                ae.W = gather(ae.W);
                ae.b = gather(ae.b);
                ae.c = gather(ae.c);
            end
            reconstruction_errors = {train_reconstruction_errors, valid_reconstruction_errors};
        end
        
        function [x_test_encode, x_test_reconstruct, test_reconstruction_error] = reconstruct(ae, data_test, test_error, output)
            x_test = single(data_test{1});
            [~, x_test_encode, ~, x_test_reconstruct] = ae.forward(x_test);
            x_test_encode = {x_test_encode};
            if test_error
                test_reconstruction_error = ae.reconstruction_error(x_test, x_test_reconstruct);
                if output
                    fprintf('Test reconstruction error      : %7.4f\n', test_reconstruction_error);
                end
            end
        end
    end
    
    methods (Access = protected)
        function ae = SGD(ae, x, alpha, l1, l2, momentum)
            n = size(x, 2);
            ae = ae.backward(x);
            ae.update_W = momentum .* ae.update_W - alpha .* (ae.nabla_W ./ n + l1 .* sign(ae.W) + l2 .* ae.W);
            ae.update_b = momentum .* ae.update_b - alpha .* (ae.nabla_b ./ n);
            ae.update_c = momentum .* ae.update_c - alpha .* (ae.nabla_c ./ n);
            ae.W = ae.W + ae.update_W;
            ae.b = ae.b + ae.update_b;
            ae.c = ae.c + ae.update_c;
        end
        
        function ae = backward(ae, x)
            [a_x, h_x, ~, x_h] = ae.forward(x);
            delta = x_h - x;
            ae.nabla_W = (delta * h_x')';
            ae.nabla_c = sum(delta, 2);
            delta = ae.W'' * delta .* ae.g_prime(a_x);
            ae.nabla_W = ae.nabla_W + delta * x';
            ae.nabla_b = sum(delta, 2);
        end
        
        function [a_x, h_x, a_h, x_h] = forward(ae, x)
            n = size(x, 2);
            x_n = x .* (rand(size(x)) > ae.dropout);
            a_x = ae.W * x_n + repmat(ae.b, 1, n);
            h_x = ae.g(a_x);
            a_h = ae.W' * h_x + repmat(ae.c, 1, n);
            x_h = ae.g(a_h);
        end
    end
    
    methods (Access = protected, Static)
        function error = reconstruction_error(x, x_reconstruct)
            error = sum(mean((x_reconstruct - x) .^ 2, 1), 2) / size(x, 2);
        end
    end
end
