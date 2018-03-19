% Neural Networks
classdef NN
    properties
        sizes
        g
        g_prime
        o
        W
        b
        batch_norm
        batch_size
        dropout
    end
    
    properties (Access = protected)
        nabla_W
        nabla_b
        update_W
        update_b
    end
    
    methods
        function nn = NN(sizes, g_type)
            nn.sizes = sizes;
            switch g_type
                case 'linear'
                    a = 1;
                    nn.g = @(x) x;
                    nn.g_prime = @(x) ones(size(x), 'single');
                case 'sigmoid'
                    a = 1;
                    nn.g = @(x) 1 ./ (1 + exp(-x));
                    nn.g_prime = @(x) nn.g(x) .* (1 - nn.g(x));
                case 'tanh'
                    a = 1;
                    nn.g = @tanh;
                    nn.g_prime = @(x) 1 - tanh(x) .^ 2;
                case 'relu'
                    a = 2;
                    nn.g = @(x) max(x, zeros(size(x), 'single'));
                    nn.g_prime = @(x) single(x > 0);
            end
            nn.o = @nn.softmax;
            nn.W = cell(1, size(nn.sizes, 2) - 1);
            nn.b = cell(1, size(nn.sizes, 2) - 1);
            nn.nabla_W = cell(1, size(nn.sizes, 2) - 1);
            nn.nabla_b = cell(1, size(nn.sizes, 2) - 1);
            nn.update_W = cell(1, size(nn.sizes, 2) - 1);
            nn.update_b = cell(1, size(nn.sizes, 2) - 1);
            rng('default');
            for i = 1:size(nn.sizes, 2) - 1
                nn.W{i} = randn(nn.sizes(i + 1), nn.sizes(i), 'single') .* sqrt(a / nn.sizes(i));
                nn.b{i} = zeros(nn.sizes(i + 1), 1, 'single');
                nn.nabla_W{i} = zeros(nn.sizes(i + 1), nn.sizes(i), 'single');
                nn.nabla_b{i} = zeros(nn.sizes(i + 1), 1, 'single');
                nn.update_W{i} = zeros(nn.sizes(i + 1), nn.sizes(i), 'single');
                nn.update_b{i} = zeros(nn.sizes(i + 1), 1, 'single');
            end
            nn.batch_norm = false;
            nn.batch_size = 1;
            nn.dropout = 0;
        end
        
        function [nn, cross_entropy_errors, classification_errors] = train(nn, data_train, data_valid, batch_norm, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output)
            x_train = single(data_train{1});
            y_train = single(data_train{2});
            n_train = size(y_train, 2);
            x_valid = single(data_valid{1});
            y_valid = single(data_valid{2});
            nn.batch_norm = batch_norm;
            nn.batch_size = batch_size;
            nn.dropout = dropout;
            train_cross_entropy_errors = zeros(epoch, 1, 'single');
            valid_cross_entropy_errors = zeros(epoch, 1, 'single');
            train_classification_errors = zeros(epoch, 1, 'single');
            valid_classification_errors = zeros(epoch, 1, 'single');
            if use_gpu
                x_train = gpuArray(x_train);
                y_train = gpuArray(y_train);
                x_valid = gpuArray(x_valid);
                y_valid = gpuArray(y_valid);
                train_cross_entropy_errors = gpuArray(train_cross_entropy_errors);
                valid_cross_entropy_errors = gpuArray(valid_cross_entropy_errors);
                train_classification_errors = gpuArray(train_classification_errors);
                valid_classification_errors = gpuArray(valid_classification_errors);
                for i = 1:size(nn.sizes, 2) - 1
                    nn.W{i} = gpuArray(nn.W{i});
                    nn.b{i} = gpuArray(nn.b{i});
                    nn.nabla_W{i} = gpuArray(nn.nabla_W{i});
                    nn.nabla_b{i} = gpuArray(nn.nabla_b{i});
                    nn.update_W{i} = gpuArray(nn.update_W{i});
                    nn.update_b{i} = gpuArray(nn.update_b{i});
                end
            end
            rng('default');
            for i = 1:epoch
                shuffle = randperm(n_train);
                x_train = x_train(:, shuffle);
                y_train = y_train(:, shuffle);
                for j = 1:nn.batch_size:n_train
                    x_batch = x_train(:, j:min(j + nn.batch_size - 1, end));
                    y_batch = y_train(:, j:min(j + nn.batch_size - 1, end));
                    nn = nn.SGD(x_batch, y_batch, alpha, l1, l2, momentum);
                end
                [~, train_cross_entropy_errors(i), train_classification_errors(i)] = nn.predict({x_train, y_train}, true, false);
                [~, valid_cross_entropy_errors(i), valid_classification_errors(i)] = nn.predict({x_valid, y_valid}, true, false);
                if output
                    fprintf('Epoch %d completed\n', i);
                    fprintf('Training cross-entropy error  : %6.4f  Training classification error  : %5.2f%%\n', train_cross_entropy_errors(i), train_classification_errors(i));
                    fprintf('Validation cross-entropy error: %6.4f  Validation classification error: %5.2f%%\n', valid_cross_entropy_errors(i), valid_classification_errors(i));
                end
            end
            for i = 1:size(nn.sizes, 2) - 1
                nn.nabla_W{i} = zeros(nn.sizes(i + 1), nn.sizes(i), 'single');
                nn.nabla_b{i} = zeros(nn.sizes(i + 1), 1, 'single');
                nn.update_W{i} = zeros(nn.sizes(i + 1), nn.sizes(i), 'single');
                nn.update_b{i} = zeros(nn.sizes(i + 1), 1, 'single');
            end
            if use_gpu
                train_cross_entropy_errors = gather(train_cross_entropy_errors);
                valid_cross_entropy_errors = gather(valid_cross_entropy_errors);
                train_classification_errors = gather(train_classification_errors);
                valid_classification_errors = gather(valid_classification_errors);
                for i = 1:size(nn.sizes, 2) - 1
                    nn.W{i} = gather(nn.W{i});
                    nn.b{i} = gather(nn.b{i});
                end
            end
            cross_entropy_errors = {train_cross_entropy_errors, valid_cross_entropy_errors};
            classification_errors = {train_classification_errors, valid_classification_errors};
        end
        
        function [y_test_predict, test_cross_entropy_error, test_classification_error] = predict(nn, data_test, test_error, output)
            x_test = single(data_test{1});
            h_x = nn.forward(x_test, true);
            y_test_predict = h_x{end};
            if test_error
                y_test = single(data_test{2});
                test_cross_entropy_error = nn.cross_entropy_error(y_test, y_test_predict);
                test_classification_error = nn.classification_error(y_test, y_test_predict);
                if output
                    fprintf('Test cross-entropy error      : %6.4f  Test classification error      : %5.2f%%\n', test_cross_entropy_error, test_classification_error);
                end
            end
        end
    end
    
    methods (Access = protected)
        function nn = SGD(nn, x, y, alpha, l1, l2, momentum)
            n = size(x, 2);
            nn = nn.backward(x, y);
            for i = 1:size(nn.sizes, 2) - 1
                nn.update_W{i} = momentum .* nn.update_W{i} - alpha .* (nn.nabla_W{i} ./ n + l1 .* sign(nn.W{i}) + l2 .* nn.W{i});
                nn.update_b{i} = momentum .* nn.update_b{i} - alpha .* (nn.nabla_b{i} ./ n);
                nn.W{i} = nn.W{i} + nn.update_W{i};
                nn.b{i} = nn.b{i} + nn.update_b{i};
            end
        end
        
        function [nn, delta] = backward(nn, x, y)
            [h_x, a_x, mask] = nn.forward(x, false);
            delta = h_x{size(nn.sizes, 2)} - y;
            nn.nabla_W{size(nn.sizes, 2) - 1} = delta * h_x{size(nn.sizes, 2) - 1}';
            nn.nabla_b{size(nn.sizes, 2) - 1} = sum(delta, 2);
            for i = size(nn.sizes, 2) - 2:-1:1
                delta = nn.W{i + 1}' * delta .* mask{i} .* nn.g_prime(a_x{i});
                nn.nabla_W{i} = delta * h_x{i}';
                nn.nabla_b{i} = sum(delta, 2);
            end
            delta = nn.W{1}' * delta;
        end
        
        function [h_x, a_x, mask] = forward(nn, x, predict)
            n = size(x, 2);
            h_x = cell(1, size(nn.sizes, 2));
            h_x{1} = x;
            a_x = cell(1, size(nn.sizes, 2) - 1);
            mask = cell(1, size(nn.sizes, 2) - 1);
            for i = 1:size(nn.sizes, 2) - 1
                a_x{i} = nn.W{i} * h_x{i} + repmat(nn.b{i}, 1, n);
                if i < size(nn.sizes, 2) - 1
                    if predict
                        if nn.batch_norm && nn.batch_size > 1
                            a_x{i} = a_x{i};
                        else
                            a_x{i} = a_x{i};
                        end
                        h_x{i + 1} = nn.g(a_x{i}) .* (1 - nn.dropout);
                    else
                        if nn.batch_norm && nn.batch_size > 1
                            a_x{i} = a_x{i};
                        else
                            a_x{i} = a_x{i};
                        end
                        mask{i} = rand(size(a_x{i})) > nn.dropout;
                        h_x{i + 1} = nn.g(a_x{i}) .* mask{i};
                    end
                else
                    h_x{i + 1} = nn.o(a_x{i});
                end
            end
        end
    end
    
    methods (Access = protected, Static)
        function y = softmax(x)
            n = size(x, 1);
            x_max = repmat(max(x, [], 1), n, 1);
            y = exp(x - x_max) ./ repmat(sum(exp(x - x_max), 1), n, 1);
        end
        
        function error = cross_entropy_error(y, y_predict)
            epsilon = 1e-10;
            error = sum(sum(-y .* log(y_predict + epsilon) - (1 - y) .* log(1 - y_predict + epsilon), 1), 2) / size(y, 2);
        end
        
        function error = classification_error(y, y_predict)
            [~, y] = max(y, [], 1);
            [~, y_predict] = max(y_predict, [], 1);
            error = sum(y ~= y_predict, 2) / size(y, 2) * 100;
        end
    end
end
