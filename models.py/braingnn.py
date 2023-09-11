from ..params import Param


class BrainGNN():
    dataparams = [Param("batchsize", 32, int), Param("train_split", 0.60, float), 
                  Param("validation_split", 0.2, float), Param("test_split", 0.2, float)]
    trainparams = [Param("lr", 0.01, float, optimizable=True), Param("epochs", 100, int), Param("k-folds", 0, int),
              Param("weightdecay", 5e-3, float, optimizable=True), Param("gamma", 0.5, float, optimizable = True), 
              Param("lr_sceduler_stepsize", 20, int, optimizable = True, description="decay the learning rate by gamma every <lr_sceduler_stepsize> epochs"),
                Param("lambda1", 0.1, float, optimizable=True), Param("lambda2", 0.1, float, optimizable=True)]
    architecture_params = [Param("n_GNN_layers", 2, int, optimizable=True), Param("pooling_ratio", 0.5, float, optimizable=True)]
    params = {"data":dataparams, "train":trainparams, "architecture": architecture_params}

class BrainGB():
    dataparams = [Param("batchsize", 32, int), Param("train_split", 0.60, float), 
                  Param("validation_split", 0.2, float), Param("test_split", 0.2, float)]
    trainparams = [Param("lr", 0.01, float, optimizable=True), Param("epochs", 100, int, optimizable=True), Param("k-folds", 0, int), Param("dropout", 0.5, float),
                   Param("weightdecay", 1e-4, float, optimizable=True)]
    architecture_params = [Param("n_gnn_layers", 2, int, optimizable=True), Param("n_MLP_layers", 1, int, optimizable=True), 
                           Param("hidden_dim", 256, int, optimizable=True), Param("edge_emb_dim", 256, int, optimizable=True)]
    params = {"data":dataparams, "train":trainparams, "architecture": architecture_params}