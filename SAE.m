% Stacked Autoencoders
classdef SAE
    properties
        sizes
        g_type
        ae
        nn
    end
    
    methods
        function sae = SAE(sizes, g_type)
            sae.sizes = sizes;
            sae.g_type = g_type;
            sae.ae = cell(1, size(sae.sizes, 2) - 2);
            for i = 1:size(sae.sizes, 2) - 2
                sae.ae{i} = AE(sae.sizes(i:i + 1), sae.g_type);
            end
            sae.nn = NN(sae.sizes, sae.g_type);
        end
        
        function [sae, reconstruction_errors] = pretrain(sae, data_train, data_valid, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output)
            [sae.ae{1}, reconstruction_errors] = sae.ae{1}.train(data_train, data_valid, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output);
            sae.nn.W{1} = sae.ae{1}.W;
            for i = 2:size(sae.sizes, 2) - 2
                data_train = sae.ae{i - 1}.reconstruct(data_train, false, false);
                data_valid = sae.ae{i - 1}.reconstruct(data_valid, false, false);
                [sae.ae{i}, reconstruction_errors] = sae.ae{i}.train(data_train, data_valid, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output);
                sae.nn.W{i} = sae.ae{i}.W;
            end
        end
        
        function [sae, cross_entropy_errors, classification_errors] = finetune(sae, data_train, data_valid, batch_norm, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output)
            [sae.nn, cross_entropy_errors, classification_errors] = sae.nn.train(data_train, data_valid, batch_norm, batch_size, dropout, epoch, alpha, l1, l2, momentum, use_gpu, output);
        end
        
        function sae = reset_nn(sae)
            sae.nn = NN(sae.sizes, sae.g_type);
        end
        
        function sae = set_nn(sae)
            for i = 1:size(sae.sizes, 2) - 2
                sae.nn.W{i} = sae.ae{i}.W;
            end
        end
        
        function [data_test_encode, x_test_reconstruct, test_reconstruction_error] = reconstruct(sae, data_test, test_error, output)
            for i = 1:size(sae.sizes, 2) - 3
                [data_test, x_test_reconstruct, test_reconstruction_error] = sae.ae{i}.reconstruct(data_test, test_error, output);
            end
            [data_test_encode, x_test_reconstruct, test_reconstruction_error] = sae.ae{end}.reconstruct(data_test, test_error, output);
        end
        
        function [y_test_predict, test_cross_entropy_error, test_classification_error] = predict(sae, data_test, test_error, output)
            [y_test_predict, test_cross_entropy_error, test_classification_error] = sae.nn.predict(data_test, test_error, output);
        end
    end
end
