import anndata as ad
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import argparse
import os.path as osp
from pprint import pformat
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, load_npz
from sklearn.model_selection import train_test_split
from dance.utils import set_seed
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import config
from utils.posenc import laplacian_positional_encoding
from graph_construct import create_graph
from scmoformer import ScMoFormer


logger = config.get_logger("train")

parser = argparse.ArgumentParser()
parser.add_argument("--subtask", type=str, default='cite')
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--max_epoch", type=int, default=5000)
parser.add_argument("--n_comp", type=int, default=128)
parser.add_argument("--pos_enc", type=str, default='rw') # ['lap', 'rw']
args = parser.parse_args()
device = args.device
subtask = args.subtask
n_comp = args.n_comp
print(pformat(vars(args)))

def CorrLoss(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c, 1) / torch.sqrt(torch.sum(y_true_c * y_true_c, 1)) / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1)))
    return -pearson

# Read files according to subtask
if subtask == 'cite':
    par = {
        "feat_path": osp.join(config.PROCESSED_DATA_DIR, 'svd', f'train_{subtask}_inputs_svd{n_comp}.h5ad'),
        "gene_adj_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_aligned_{subtask}.npz'),
        "input_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_inputs.h5ad'),
        "target_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_targets.h5ad'),
        "symbols_path": osp.join(config.PROCESSED_DATA_DIR, f'gene_protein_edges_{subtask}.h5'),
        "ppi_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_target_{subtask}.npz'),
        "output_path": osp.join(config.OUT_DIR,  f'scmoformer_{subtask}'),
    }
    meta = ad.read_h5ad(par["input_path"], backed="r").obs
    var = ad.read_h5ad(par["target_path"], backed="r").var
    days = sorted(meta["day"].unique())
    donors = sorted(meta["donor"].unique())
    train_idx = meta.index.get_indexer(meta.index[~(meta['day'] == days[-1])])
    test_idx = meta.index.get_indexer(meta.index[meta['day'] == days[-1]])
    meta_test = meta.iloc[test_idx]

    X = ad.read_h5ad(par["feat_path"]).to_df().iloc[:,:128]
    feat = torch.from_numpy(X.to_numpy())
    feat_dim = feat.shape[1]
    print(feat.shape)

    gene_adj = np.load(par["gene_adj_path"], allow_pickle=True)
    symbols = pd.read_hdf(par["symbols_path"]).to_numpy()
    ppi = np.load(par["ppi_path"], allow_pickle=True)
    X_input =  ad.read_h5ad(par["input_path"]).to_df()
    obs = torch.from_numpy(X_input.to_numpy())
    print(obs.shape)

    Y = ad.read_h5ad(par["target_path"]).to_df()
    num_pro = Y.shape[1]
    labels = torch.from_numpy(Y.to_numpy())
    print(labels.shape)

elif subtask == 'gex2adt':
    par = {
        "input_path": osp.join(config.PROCESSED_DATA_DIR, 'svd', f'train_{subtask}_inputs_svd{n_comp}.h5ad'),
        "test_input_path": osp.join(config.PROCESSED_DATA_DIR, 'svd', f'test_{subtask}_inputs_svd{n_comp}.h5ad'),
        "gene_adj_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_aligned_{subtask}.npz'),
        "src_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_inputs.h5ad'),
        "test_src_path": osp.join(config.RAW_DATA_DIR, f'test_{subtask}_inputs.h5ad'),
        "target_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_targets.h5ad'),
        "test_target_path": osp.join(config.RAW_DATA_DIR, f'test_{subtask}_targets.h5ad'),
        "symbols_path": osp.join(config.PROCESSED_DATA_DIR, f'gene_protein_edges_{subtask}.h5'),
        "ppi_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_target_{subtask}.npz'),
        "output_path": osp.join(config.OUT_DIR,  f'scmoformer_{subtask}'),
    }
    var = ad.read_h5ad(par["target_path"], backed="r").var
    X = ad.read_h5ad(par["input_path"]).to_df()
    X_test = ad.read_h5ad(par["test_input_path"])
    meta_test = X_test.obs
    X_test = X_test.to_df()
    train_size = len(X)
    test_size = len(X_test)
    train_idx = range(train_size)
    test_idx = range(train_size, train_size + test_size)

    X_test.columns = X.columns
    feat = torch.from_numpy(pd.concat([X, X_test]).to_numpy())
    feat_dim = feat.shape[1]
    print(feat.shape)

    gene_adj = np.load(par["gene_adj_path"], allow_pickle=True)
    symbols = pd.read_hdf(par["symbols_path"]).to_numpy()
    ppi = np.load(par["ppi_path"], allow_pickle=True)
    X_src = ad.read_h5ad(par["src_path"]).to_df()
    X_src_test = ad.read_h5ad(par["test_src_path"]).to_df()
    X_src_test.columns = X_src.columns
    obs = torch.from_numpy(pd.concat([X_src, X_src_test]).to_numpy())
    print(obs.shape)
    
    Y = ad.read_h5ad(par["target_path"]).to_df()
    Y_test = ad.read_h5ad(par["test_target_path"]).to_df()
    num_pro = Y.shape[1]
    labels = torch.from_numpy(pd.concat([Y, Y_test]).to_numpy())
    print(labels.shape)

# Determine parameter
act, sage_agg, pos_enc_dim, layers = "selu", "mean", 10, [512, 512, 512, 512]
patience, tol, lr, trans_head = 20, 40, 1e-3, 8
local_gnn_type, global_model_type, pna_degrees = "SAGE", "Performer", None
params = {
    "cite": (.5, .1, 3000, 1e-6),
    "gex2adt": (.5, .1, 10000, 1e-5),
}
feat_drop, attn_drop, batch_size, weight_decay = params[subtask]

corrs = []
mses = []
maes = []
for j, sd in enumerate(args.seeds):
    set_seed(sd)
    logger.info(f'Try {j+1}, Seed {sd}')
    if subtask == 'cite':
        train_idx, valid_idx = train_test_split(train_idx, train_size=0.8, random_state=sd)
    elif subtask == 'gex2adt':
        train_idx, valid_idx = train_test_split(train_idx, train_size=0.85, random_state=sd)
    
    # Construct graph
    num_cell = obs.shape[0]
    num_gene = obs.shape[1]
    gene_feat = torch.mm(obs.T, feat)    
    pro_feat = None
    g0 = create_graph(num_cell, num_gene, num_pro, obs, symbols, ppi, 
        feat, labels, gene_adj, gene_feat, pro_feat)
    
    # Calculate positional encoding
    g_pro = dgl.to_homogeneous(g0.edge_type_subgraph(['ppi']), ndata=['feat'], edata=['edge_weight'])
    g_gene = dgl.to_homogeneous(g0.edge_type_subgraph(['coexp']), edata=['edge_weight'])    
    pos_enc = args.pos_enc
    if pos_enc == 'lap':
        pro_lap_pos_enc = laplacian_positional_encoding(g_pro, pos_enc_dim)
        g0.nodes['protein'].data['pos_enc'] = pro_lap_pos_enc
        gene_lap_pos_enc = laplacian_positional_encoding(g_gene, pos_enc_dim)
        g0.nodes['gene'].data['pos_enc'] = gene_lap_pos_enc
    elif pos_enc == 'rw':
        pro_rw_pos_enc = dgl.random_walk_pe(g_pro, pos_enc_dim, 'edge_weight')
        g0.nodes['protein'].data['pos_enc'] = pro_rw_pos_enc
        gene_rw_pos_enc = dgl.random_walk_pe(g_gene, pos_enc_dim, 'edge_weight')
        g0.nodes['gene'].data['pos_enc'] = gene_rw_pos_enc.cpu()
    
    # Training
    train_loader = DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_idx, batch_size=batch_size, shuffle=True)
    model = ScMoFormer(dims=[num_gene] + layers + [num_pro], feat_dim=feat_dim, pos_enc=pos_enc,
        num_pro=num_pro, pos_enc_dim=pos_enc_dim, local_gnn_type=local_gnn_type, 
        global_model_type=global_model_type, trans_head=trans_head, pna_degrees=pna_degrees,
        feat_drop=feat_drop, attn_drop=attn_drop, act=act, sage_agg=sage_agg).to(device)
    mse = nn.MSELoss()
    corr = CorrLoss
    mae = nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tempcorrs = []
    tempmses = []
    tempmaes = []
    scheduler = ReduceLROnPlateau(opt, 'min', patience=patience, factor=0.9)
    counter = 0
    out_path = f"scmoformer_{subtask}_{sd}"
    graphs = {}
    for k, batch in enumerate(train_loader):
        node_dict = {
            'cell': batch.int(),
            'protein': torch.arange(num_pro).int(), 
            'gene': torch.arange(num_gene).int()
        }
        graphs[k] = dgl.node_subgraph(g0, node_dict).to(device)

    for i in tqdm(range(args.max_epoch)):
        model.train() 
        for k, batch in enumerate(train_loader):
            Z = model(graphs[k].to(device))
            loss = mse(Z, graphs[k].nodes['cell'].data['label'].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        del Z
        with torch.no_grad():
            model.eval()
            Z = model(g0.to(device))
            Y_val = g0.nodes['cell'].data['label'][valid_idx].to(device)
            valcorr = corr(Z[valid_idx], Y_val).cpu().item()
            valmse = mse(Z[valid_idx], Y_val).cpu().item()
            valmae = mae(Z[valid_idx], Y_val).cpu().item()
            tempcorrs.append(valcorr)
            tempmses.append(valmse)
            tempmaes.append(valmae)

            Y_test = g0.nodes['cell'].data['label'][test_idx].to(device)
            testcorr = corr(Z[test_idx], Y_test).cpu().item()
            testmse = mse(Z[test_idx], Y_test).cpu().item()
            testmae = mae(Z[test_idx], Y_test).cpu().item()

        val = valmse
        tempval = tempmses
        if i >= 0:
            if val > min(tempval):
                counter += 1
                logger.info(f'early stopping counter: {counter}')
            else:
                torch.save(model.state_dict(), osp.join(config.OUT_DIR, f'{out_path}.pth'))
        if counter == tol:
            logger.info(f'Early stopped. Best val corr: {-min(tempcorrs):0.8f}, \
                best val MSE: {min(tempmses):0.8f}, best val MAE: {min(tempmaes):0.8f}')
            break
        scheduler.step(val)
        logger.info(f'epoch {i + 1}, training: {loss.cpu().item():0.8f}, \
            val corr: {valcorr:0.8f}, val MSE: {valmse:0.8f}, val MAE: {valmae:0.8f} \
            test corr: {testcorr:0.8f}, test MSE: {testmse:0.8f}, test MAE: {testmae:0.8f}')
    
    model.load_state_dict(torch.load(osp.join(config.OUT_DIR, f'{out_path}.pth')))
    model = model.cpu()
    with torch.no_grad():
        model.eval()
        Z = model(g0)[test_idx]
        Y_test = g0.nodes['cell'].data['label'][test_idx]
        testcorr = corr(Z, Y_test).item()
        testmse = mse(Z, Y_test).item()
        testmae = mae(Z, Y_test).item()
        corrs.append(testcorr)
        mses.append(testmse)
        maes.append(testmae)
    logger.info(f"Try {j + 1}, corr = {testcorr:.8f}, MSE = {testmse:.8f}, MAE = {testmae:.8f}")

rmses = np.sqrt(mses)
logger.info(f'Corr: {np.mean(corrs):.5f} +/- {np.std(corrs):.5f}')
logger.info(f'MSE: {np.mean(mses):.5f} +/- {np.std(mses):.5f}')
logger.info(f'RMSE: {np.mean(rmses):.5f} +/- {np.std(rmses):.5f}')
logger.info(f'MAE: {np.mean(maes):.5f} +/- {np.std(maes):.5f}')
logger.info(f'Corrs: {corrs}')
logger.info(f'MSEs: {mses}')
logger.info(f'RMSEs: {rmses}')
logger.info(f'MAEs: {maes}')