% Deep Belief Networks
classdef DBN
    properties
        sizes
        g_type
        rbm
        nn
    end
    
    methods
        function dbn = DBN(sizes, g_type)
            dbn.sizes = sizes;
            dbn.g_type = g_type;
            dbn.rbm = cell(1, size(dbn.sizes, 2) - 2);
            switch dbn.g_type
                case {'bb', 'gb'}
                    dbn.rbm{1} = RBM(dbn.sizes(1:2), dbn.g_type);
                    for i = 2:size(dbn.sizes, 2) - 2
                        dbn.rbm{i} = RBM(dbn.sizes(i:i + 1), 'bb');
                    end
                    dbn.nn = NN(dbn.sizes, 'sigmoid');
                case {'rr', 'gr'}
                    dbn.rbm{1} = RBM(dbn.sizes(1:2), dbn.g_type);
                    for i = 2:size(dbn.sizes, 2) - 2
                        dbn.rbm{i} = RBM(dbn.sizes(i:i + 1), 'rr');
                    end
                    dbn.nn = NN(dbn.sizes, 'relu');
            end
        end
        
        function [dbn, reconstruction_errors] = pretrain(dbn, data_train, data_valid, batch_size, epoch, alpha, l1, l2, momentum, k, k_pcd, use_gpu, output)
            [dbn.rbm{1}, reconstruction_errors] = dbn.rbm{1}.train(data_train, data_valid, batch_size, epoch, alpha, l1, l2, momentum, k, k_pcd, use_gpu, output);
            dbn.nn.W{1} = dbn.rbm{1}.W{1};
            for i = 2:size(dbn.sizes, 2) - 2
                data_train = dbn.rbm{i - 1}.reconstruct(data_train, false, false);
                data_valid = dbn.rbm{i - 1}.reconstruct(data_valid, false, false);
                [dbn.rbm{i}, reconstruction_errors] = dbn.rbm{i}.train(data_train, data_valid, batch_size, epoch, alpha, l1, l2, momentum, k, k_pcd, use_gpu, output);
                dbn.nn.W{i} = dbn.rbm{i}.W{1};
            end
        end
        
        function [dbn, cross_entropy_errors, classification_errors] = finetune(dbn, data_train, data_valid, batch_norm, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output)
            [dbn.nn, cross_entropy_errors, classification_errors] = dbn.nn.train(data_train, data_valid, batch_norm, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output);
        end
        
        function dbn = reset_nn(dbn)
            switch dbn.g_type
                case {'bb', 'gb'}
                    dbn.nn = NN(dbn.sizes, 'sigmoid');
                case {'rr', 'gr'}
                    dbn.nn = NN(dbn.sizes, 'relu');
            end
        end
        
        function dbn = set_nn(dbn)
            for i = 1:size(dbn.sizes, 2) - 2
                dbn.nn.W{i} = dbn.rbm{i}.W{1};
            end
        end
        
        function [data_test_encode, x_test_reconstruct, test_reconstruction_error] = reconstruct(dbn, data_test, test_error, output)
            for i = 1:size(dbn.sizes, 2) - 3
                [data_test, x_test_reconstruct, test_reconstruction_error] = dbn.rbm{i}.reconstruct(data_test, test_error, output);
            end
            [data_test_encode, x_test_reconstruct, test_reconstruction_error] = dbn.rbm{end}.reconstruct(data_test, test_error, output);
        end
        
        function [y_test_predict, test_cross_entropy_error, test_classification_error] = predict(dbn, data_test, test_error, output)
            [y_test_predict, test_cross_entropy_error, test_classification_error] = dbn.nn.predict(data_test, test_error, output);
        end
    end
end
