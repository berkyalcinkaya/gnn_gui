from params import Param


class BrainGNN():
    dataparams = [Param("batchsize", 32, int), Param("train_split", 0.60, float), 
                  Param("validation_split", 0.2, float), Param("test_split", 0.2, float)]
    trainparams = [Param("lr", 0.01, float, optimizable=True, default_search_space=[0.001, 0.0005, 1]),
                Param("epochs", 100, int, optimizable=True, default_search_space=[50,80,100]), 
                Param("k-folds", 0, int),
              Param("weightdecay", 5e-3, float, optimizable=True), 
              Param("gamma", 0.5, float, optimizable = True), 
              Param("lr_sceduler_stepsize", 20, int, optimizable = True, description="decay the learning rate by gamma every <lr_sceduler_stepsize> epochs"),
                Param("lambda1", 0.1, float, optimizable=True), 
                Param("lambda2", 0.1, float, optimizable=True)]
    architecture_params = [Param("n_GNN_layers", 2, int, optimizable=True), 
                           Param("pooling_ratio", 0.5, float, optimizable=True)]
    params = {"data":dataparams, "train":trainparams, "architecture": architecture_params}

class BrainGB():
    dataparams = [Param("batchsize", 32, int), Param("train_split", 0.60, float), 
                  Param("validation_split", 0.2, float), Param("test_split", 0.2, float)]
    trainparams = [Param("lr", 0.01, float, optimizable=True, default_search_space=[0.1, 0.01, 0.001, 0.0001]), 
                   Param("epochs", 100, int, optimizable=True, default_search_space=[50,80,100]),
                   Param("weightdecay", 1e-4, float, optimizable=True, default_search_space= [1e-5, 1e-4, 1e-3]),
                   Param("k-folds", 0, int), Param("dropout", 0.5, float)]
    architecture_params = [Param("n_GNN_layers", 2, int, optimizable=True, default_search_space=[1, 2, 3, 4]), 
                           Param("n_MLP_layers", 1, int, optimizable=True, default_search_space=[1, 2, 3, 4]), 
                           Param("hidden_dim", 256, int, optimizable=True, default_search_space=[8, 12, 16, 32]), 
                           Param("edge_emb_dim", 256, int, optimizable=True, default_search_space=[32, 64, 96, 128, 256, 512, 1024])]
    params = {"data":dataparams, "train":trainparams, "architecture": architecture_params}