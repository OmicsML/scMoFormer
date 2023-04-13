import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding with Laplacian eigenvectors 
        adapted from 
        
        "A Generalization of Transformer Networks to Graphs"
        Dwivedi, Vijay Prakash and Bresson, Xavier, 2021
        https://github.com/graphdeeplearning/graphtransformer
    """
    graph_edata = (g.edata['edge_weight'], g.edges())
    shape = (g.num_nodes(), g.num_nodes())
    A = csr_matrix(coo_matrix(graph_edata, shape))
    N = sp.diags(np.array(A.sum(0))[0] ** -0.5, dtype=float)
    L = sp.eye(A.shape[0]) - N * A * N

    EigVal, EigVec = sp.linalg.eigs(L, pos_enc_dim + 1, which="SM")
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    return torch.from_numpy(EigVec[:,1:]).float()