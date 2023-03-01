import anndata as ad
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import argparse
import os.path as osp
import ipdb
from pprint import pformat
import logging
from copy import deepcopy
import scipy.sparse as sp
from performer_pytorch import Performer
from scipy.sparse import csr_matrix, coo_matrix, load_npz
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import dgl
import dgl.nn as dglnn
from dgl import AddReverse
from dgl.nn import LabelPropagation

from dance.utils import set_seed
from dance.utils import SimpleIndexDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformer import Transformer
from layers import GPSLayer

from utils import config
from utils.posenc import laplacian_positional_encoding, wl_positional_encoding

parser = argparse.ArgumentParser()
parser.add_argument("--subtask", type=str, default='cite')
parser.add_argument("--seeds", type=int, nargs="+", default=[42])
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--max_epoch", type=int, default=5000,
                    help="number of training epochs")
parser.add_argument("--pretrain", action='store_true', default=False)
parser.add_argument("--init_pro", action='store_true', default=False)
parser.add_argument("--log_id", type=str, default='')
parser.add_argument("--run_id", type=str, default='')
parser.add_argument("--pos_enc", type=str, default='lap') # ['lap', 'rw', 'wl']
parser.add_argument("--sub", action='store_true', default=False)
parser.add_argument("--gene_mod", type=str, default='sage')
parser.add_argument("--pro_mod", type=str, default='gps')
parser.add_argument("--cell_mod", type=str, default='trans')
parser.add_argument("--agg", type=str, default='sum')
args = parser.parse_args()
print(pformat(vars(args)))

logging.basicConfig(filename=f'logs/gnn_trans_trans_batch_{args.subtask}_{args.log_id}_{args.run_id}.log',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

def CorrLoss(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c, 1) / torch.sqrt(torch.sum(y_true_c * y_true_c, 1)) / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1)))
    return -pearson

def create_graph(num_cell, num_gene, num_pro, X_prime, symbols, ppi, cell_feat, Y, 
                 gene_adj=None, gene_feat=None, pro_feat=None, training_edge=False, pro_sub_idx=None):
    # Create training graph
    obs = coo_matrix(X_prime)
    rev_obs = coo_matrix(X_prime.T)
    symbols = coo_matrix(symbols)
    rev_symbols = coo_matrix(symbols.T)
    labels = coo_matrix(Y)
    rev_labels = coo_matrix(Y.T)

    if gene_adj is not None:
        data_dict = {
            ('protein', 'ppi', 'protein'): (torch.from_numpy(ppi['head']).int(), torch.from_numpy(ppi['tail']).int()),
            ('protein', 'rev_symbol', 'gene'): (torch.from_numpy(rev_symbols.row).int(), torch.from_numpy(rev_symbols.col).int()),
            ('gene', 'symbol', 'protein'): (torch.from_numpy(symbols.row).int(), torch.from_numpy(symbols.col).int()),
            ('cell', 'obs', 'gene'): (torch.from_numpy(obs.row).int(), torch.from_numpy(obs.col).int()),
            ('gene', 'rev_obs', 'cell'): (torch.from_numpy(rev_obs.row).int(), torch.from_numpy(rev_obs.col).int()),
            ('gene', 'coexp', 'gene'): (torch.from_numpy(gene_adj['head']).int(), torch.from_numpy(gene_adj['tail']).int()),
        }
        if training_edge:
            data_dict[('cell', 'label', 'protein')] = (torch.from_numpy(labels.row).int(), torch.from_numpy(labels.col).int())
            data_dict[('protein', 'rev_label', 'cell')] = (torch.from_numpy(rev_labels.row).int(), torch.from_numpy(rev_labels.col).int())
        
        num_nodes_dict = {
            'protein': num_pro,
            'cell': num_cell,
            'gene': num_gene,
        }
    else:
        pass

    g = dgl.heterograph(data_dict, num_nodes_dict)
    g.edges['obs'].data['edge_weight'] = torch.from_numpy(obs.data).float()
    g.edges['rev_obs'].data['edge_weight'] = torch.from_numpy(rev_obs.data).float()
    g.edges['coexp'].data['edge_weight'] = torch.from_numpy(gene_adj['weight']).float()
    g.edges['ppi'].data['edge_weight'] = torch.from_numpy(ppi['weight']).float()
    g.edges['symbol'].data['edge_weight'] = torch.from_numpy(symbols.data).float()
    g.edges['rev_symbol'].data['edge_weight'] = torch.from_numpy(rev_symbols.data).float()
    if training_edge:
        g.edges['label'].data['edge_weight'] = torch.from_numpy(labels.data).float()
        g.edges['rev_label'].data['edge_weight'] = torch.from_numpy(rev_labels.data).float()
    
    if pro_sub_idx is not None:
        node_dict = {
            'protein': torch.tensor(pro_sub_idx).int(),
            'cell': torch.arange(num_cell).int(), 
            'gene': torch.arange(num_gene).int()
        }
        g = dgl.node_subgraph(g, node_dict, store_ids=False)

    g.nodes['cell'].data['feat'] = cell_feat.cpu().float()
    if gene_feat is not None:
        g.nodes['gene'].data['feat'] = gene_feat.cpu().float()
    else:
        g.nodes['gene'].data['feat'] = torch.arange(g.num_nodes('gene')).long()
    if pro_feat is not None:
        g.nodes['protein'].data['feat'] = pro_feat.cpu().float()
    else:
        g.nodes['protein'].data['feat'] = torch.arange(g.num_nodes('protein')).long()
    g.nodes['cell'].data['label'] = Y.cpu().float()
    return g

class ScMoFormer(nn.Module):
    def __init__(self, dims, feat_dim, num_cell, num_gene, num_pro, pos_enc, pos_enc_dim=10, local_gnn_type='SAGE', 
        global_model_type='Performer', gat_head=4, trans_head=8, pna_degrees=None, feat_drop=0.1, attn_drop=0.1, 
        gene_mod='sage', pro_mod='gps', agg='sum', act='selu', sage_agg='mean', cell_mod='trans'):
        super().__init__()
        self.layers = len(dims) - 1
        self.pro_mod = pro_mod
        self.gene_mod = gene_mod
        self.cell_mod = cell_mod
        self.pos_enc = pos_enc
        self.agg = agg

        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'selu':
            self.act = nn.SELU()
        self.dp = nn.Dropout(feat_drop)

        self.gatconvs = nn.ModuleList()
        self.sageconvs = nn.ModuleList()
        self.pro_gps = nn.ModuleList()
        self.pro_sage = nn.ModuleList()
        self.gene_gps = nn.ModuleList()
        self.gene_sage = nn.ModuleList()
        self.pro_trans = nn.ModuleList()
        self.gene_trans = nn.ModuleList()
        self.cell_trans = nn.ModuleList()
        
        self.mlpcell = nn.ModuleList()
        self.mlpgene = nn.ModuleList()
        self.mlpprotein = nn.ModuleList()
        self.mapcell = nn.ModuleList()
        self.mapgene = nn.ModuleList()
        self.mapprotein = nn.ModuleList()
        self.edges = ['obs', 'rev_obs', 'coexp', 'ppi', 'symbol', 'rev_symbol']
        self.edges_sub = ['obs', 'rev_obs', 'symbol', 'rev_symbol']
        self.nodes = ['gene', 'protein', 'cell']
        for i in range(1, self.layers-1):
            self.gatconvs.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [
                dglnn.GATConv(dims[i], dims[i + 1], gat_head, feat_drop=feat_drop, attn_drop=attn_drop, residual=True)
                for j in range(len(self.edges))])), aggregate='sum'))        
            self.sageconvs.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [
                dglnn.SAGEConv(dims[i], dims[i + 1], aggregator_type=sage_agg, feat_drop=feat_drop) 
                for j in range(len(self.edges))])), aggregate='stack'))
            self.pro_gps.append(
                GPSLayer(dims[i], local_gnn_type, global_model_type, trans_head, self.act, pna_degrees, 
                            dropout=feat_drop, attn_dropout=attn_drop)) #, log_attn_weights=True))
            self.pro_sage.append(
                dglnn.SAGEConv(dims[i], dims[i + 1], aggregator_type=sage_agg, feat_drop=feat_drop))
            self.gene_gps.append(
                GPSLayer(dims[i], local_gnn_type, global_model_type, trans_head, self.act, pna_degrees, 
                            dropout=feat_drop, attn_dropout=attn_drop)) #, log_attn_weights=True)
            self.gene_sage.append(
                dglnn.SAGEConv(dims[i], dims[i + 1], aggregator_type=sage_agg, feat_drop=feat_drop))
            self.gene_trans.append(
                Performer(dims[i], depth=1, heads=trans_head, dim_head=int(dims[i] / trans_head), attn_dropout=attn_drop,
                          ff_dropout=feat_drop, ff_mult=2, kernel_fn=self.act, causal=False))
            self.pro_trans.append(
                Performer(dims[i], depth=1, heads=trans_head, dim_head=int(dims[i] / trans_head), attn_dropout=attn_drop,
                          ff_dropout=feat_drop, ff_mult=2, kernel_fn=self.act, causal=False))
            self.cell_trans.append(
                Performer(dims[i], depth=1, heads=trans_head, dim_head=int(dims[i] / trans_head), attn_dropout=attn_drop,
                          ff_dropout=feat_drop, ff_mult=2, kernel_fn=self.act, causal=False))
            self.mlpcell.append(nn.Linear(dims[i], dims[i + 1]))
            self.mapprotein.append(nn.Linear(dims[i] + dims[i + 1], dims[i + 1]))
            self.mapgene.append(nn.Linear(dims[i] + 2 * dims[i + 1], dims[i + 1]))
            if self.cell_mod == 'trans':
                self.mapcell.append(nn.Linear(dims[i] + 2 * dims[i + 1], dims[i + 1]))
            else:
                self.mapcell.append(nn.Linear(2 * dims[i + 1], dims[i + 1]))
            
        
        self.emb_cell = nn.Linear(feat_dim, dims[1])
        self.emb_gene = nn.Linear(feat_dim, dims[1])
        if args.pretrain or args.init_pro:
            self.emb_protein = nn.Linear(feat_dim, dims[1])
        else:
            self.emb_protein = nn.Embedding(num_pro, dims[1])
        
        if pos_enc in ['rw', 'lap'] :
            self.emb_gene_pos_enc = nn.Linear(pos_enc_dim, dims[1])
            self.emb_pro_pos_enc = nn.Linear(pos_enc_dim, dims[1])
        elif pos_enc == 'wl':
            self.emb_gene_pos_enc = nn.Embedding(num_gene, dims[1])
            self.emb_pro_pos_enc = nn.Embedding(num_pro, dims[1])

        self.pro_gate = nn.Parameter(torch.ones(num_pro, 2))
        self.gene_gate = nn.Parameter(torch.ones(num_gene, 3))
        if self.cell_mod == 'trans':
            self.cell_gate = nn.Parameter(torch.ones(num_cell, 3))
        else:
            self.cell_gate = nn.Parameter(torch.ones(num_cell, 2))

        self.mlp_last = nn.Linear(sum(dims[1:-1]), dims[-1])
        self.conv_norm1 = nn.ModuleList()
        self.conv_norm2 = nn.ModuleList()
        for i in range((self.layers-2) * len(self.nodes)):
            self.conv_norm1.append(nn.LayerNorm(dims[i // len(self.nodes) + 2]))
            self.conv_norm2.append(nn.LayerNorm(dims[i // len(self.nodes) + 2]))

    def conv(self, graph, layer, h):
        # hgene = self.conv_norm1[layer * len(self.nodes)](self.act(self.mlpgene[layer](h['gene'])))
        # hprotein = self.conv_norm1[layer * len(self.nodes) + 1](self.act(self.mlpprotein[layer](h['protein'])))
        hmlp_cell = self.conv_norm1[layer * len(self.nodes) + 2](self.act(self.mlpcell[layer](h['cell'])))
        
        g_gene = dgl.to_homogeneous(graph.edge_type_subgraph(['coexp']), edata=['edge_weight'])
        g_pro = dgl.to_homogeneous(graph.edge_type_subgraph(['ppi']), edata=['edge_weight'])
        g_sub = graph.edge_type_subgraph(self.edges_sub)
        hsage = self.sageconvs[layer](g_sub, h, mod_kwargs=dict(zip(self.edges, [{'edge_weight':
            self.dp(graph.edges['obs'].data['edge_weight'])}] + [{'edge_weight': 
            self.dp(graph.edges['rev_obs'].data['edge_weight'])}] + [{'edge_weight':
            graph.edges[self.edges_sub[i]].data['edge_weight']} for i in range(2, len(self.edges_sub))])))

        if self.pro_mod == 'gps':
            hpro_pro = self.pro_gps[layer](g_pro, h['protein'], graph.edges['ppi'].data['edge_weight'])
        elif self.pro_mod == 'sage':
            hpro_pro = self.pro_sage[layer](g_pro, h['protein'], graph.edges['ppi'].data['edge_weight'])
        elif self.pro_mod == 'trans':
            hpro_pro = self.pro_trans[layer](h['protein'].unsqueeze(0))[-1,:,:]
        
        if self.gene_mod == 'gps':
            hgene_gene = self.gene_gps[layer](g_gene, h['gene'], graph.edges['coexp'].data['edge_weight'])
        elif self.gene_mod == 'sage':
            hgene_gene = self.gene_sage[layer](g_gene, h['gene'], graph.edges['coexp'].data['edge_weight'])
        elif self.gene_mod == 'trans':
            hgene_gene = self.gene_trans[layer](h['gene'].unsqueeze(0))[-1,:,:]

        if self.cell_mod == 'trans':
            hcell_cell = self.cell_trans[layer](h['cell'].unsqueeze(0))[-1,:,:]
            hcell_cell = self.conv_norm1[layer * len(self.nodes) + 2](self.act(hcell_cell))
        
        hgene_gene = self.conv_norm1[layer * len(self.nodes)](self.act(hgene_gene))
        hpro_pro = self.conv_norm1[layer * len(self.nodes) + 1](self.act(hpro_pro))
        hsage['gene'] = self.conv_norm1[layer * len(self.nodes)](self.act(hsage['gene']))
        hsage['protein'] = self.conv_norm1[layer * len(self.nodes) + 1](self.act(hsage['protein']))
        hsage['cell'] = self.conv_norm1[layer * len(self.nodes) + 2](self.act(hsage['cell']))

        if self.agg == 'sum':
            hpro = hsage['protein'].sum(1) + hpro_pro
            hgen = hsage['gene'].sum(1) + hgene_gene
            if self.cell_mod == 'trans':
                hcel = hsage['cell'].sum(1) + hmlp_cell + hcell_cell
            else:
                hcel = hsage['cell'].sum(1) + hmlp_cell

        elif self.agg == 'cat':
            hsage['protein'] = hsage['protein'].reshape(hsage['protein'].shape[0], -1)
            hpro = self.mapprotein[layer](torch.cat([hpro_pro, hsage['protein']], 1))
            hsage['gene'] = hsage['gene'].reshape(hsage['gene'].shape[0], -1)
            hgen = self.mapgene[layer](torch.cat([hgene_gene, hsage['gene']], 1))
            hsage['cell'] = hsage['cell'].reshape(hsage['cell'].shape[0], -1)
            if self.cell_mod == 'trans':
                hcel = self.mapcell[layer](torch.cat([hcell_cell, hmlp_cell, hsage['gene']], 1))
            else:
                hcel = self.mapcell[layer](torch.cat([hmlp_cell, hsage['gene']], 1))
            
        elif self.agg == 'gate':
            hpro = torch.cat([hpro_pro.unsqueeze(-1), hsage['protein'].permute(0,2,1)], -1)
            hpro = torch.einsum('ijk,ik->ij', hpro, self.pro_gate)
            hgen = torch.cat([hgene_gene.unsqueeze(-1), hsage['gene'].permute(0,2,1)], -1)
            hgen = torch.einsum('ijk,ik->ij', hgen, self.gene_gate)
            if self.cell_mod == 'trans':
                hcel = torch.cat([hcell_cell.unsqueeze(-1), hmlp_cell.unsqueeze(-1), 
                                  hsage['cell'].permute(0,2,1)], -1)
            else:
                hcel = torch.cat([hmlp_cell.unsqueeze(-1), hsage['cell'].permute(0,2,1)], -1)
            hcel = torch.einsum('ijk,ik->ij', hcel, self.cell_gate)
        
        hgene = self.conv_norm2[layer * len(self.nodes)](self.act(hgen))
        hprotein = self.conv_norm2[layer * len(self.nodes) + 1](self.act(hpro))
        hcell = self.conv_norm2[layer * len(self.nodes) + 2](self.act(hcel))
        return {'gene': hgene, 'protein': hprotein, 'cell': hcell}

    def forward(self, graph):
        hpro = self.emb_protein(graph.nodes['protein'].data['feat'])
        if self.pro_mod in ['gps', 'trans'] and self.pos_enc in ['lap', 'rw', 'wl']:
            hpro += self.emb_pro_pos_enc(graph.nodes['protein'].data['pos_enc'])

        hgene = self.emb_gene(graph.nodes['gene'].data['feat'])
        if self.gene_mod in ['gps', 'trans'] and self.pos_enc in ['lap', 'rw', 'wl']:
            hgene += self.emb_gene_pos_enc(graph.nodes['gene'].data['pos_enc'])
        h0 = {
            'cell': self.dp(self.act(self.emb_cell(graph.nodes['cell'].data['feat']))),
            'gene': self.dp(self.act(hgene)),
            'protein': self.dp(self.act(hpro)),
        }
        h = h0.copy()
        cell_hist = []
        cell_hist.append(h0['cell'])
        for i in range(1, self.layers-1):            
            h = self.conv(graph, i - 1, h)
            cell_hist.append(h['cell'])

        pred = self.mlp_last(self.dp(torch.cat(cell_hist, 1)))
        return pred

class GAE(nn.Module):
    def __init__(self, dims, num_nodes):
        super().__init__()
        self.layers = len(dims) - 1
        self.dims = dims
        self.emb = nn.Embedding(num_nodes, dims[0])
        self.act = nn.Sigmoid()
        self.dp = nn.Dropout(0.1)
        self.sageconvs = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        for i in range(self.layers):
            self.sageconvs.append(dglnn.SAGEConv(dims[i], dims[i + 1], aggregator_type='mean', feat_drop=0.1))
            self.conv_norm.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, graph):
        h = self.emb(graph.ndata['feat'])
        for i in range(self.layers):  
            h = self.sageconvs[i](graph, h, graph.edata['edge_weight'])
            h = self.conv_norm[i](h)
        h = self.dp(h)
        adj = self.act(torch.mm(h, h.t()))
        return h, adj

device = args.device
subtask = args.subtask
if subtask == 'cite':
    par = {
        "feat_path": osp.join(config.PROCESSED_DATA_DIR, 'svd', f'train_{subtask}_inputs_svd128.h5ad'),
        "gene_adj_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_aligned_{subtask}.npz'),
        "src_path": osp.join(config.RAW_DATA_DIR, 'raw', f'train_{subtask}_inputs_raw_aligned.h5ad'),
        "sct_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_inputs_SCT_11_14_2022.h5ad'),
        "input_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_inputs.h5ad'),
        "target_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_targets.h5ad'),
        "raw_target_path": osp.join(config.RAW_DATA_DIR, 'raw', f'train_{subtask}_targets_raw.h5ad'),
        "meta_path": osp.join(config.RAW_DATA_DIR, 'train_cite_inputs.h5ad'),
        "var_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_targets.h5ad'),
        "symbols_path": osp.join(config.PROCESSED_DATA_DIR, f'gene_protein_edges_{subtask}.h5'),
        "ppi_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_target_{subtask}.npz'),
        "output_path": osp.join(config.OUT_DIR,  f'gnn_trans_trans_batch_{subtask}'),
    }
    meta = ad.read_h5ad(par["meta_path"], backed="r").obs
    var = ad.read_h5ad(par["var_path"], backed="r").var
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
    overlapped_idx = sorted(set(list(ppi['head']) + list(ppi['tail'])))
    X_src = ad.read_h5ad(par["src_path"]).to_df()
    X_sct = ad.read_h5ad(par["sct_path"]).to_df()
    X_input =  ad.read_h5ad(par["input_path"]).to_df()
    obs = torch.from_numpy(X_input.to_numpy())
    print(obs.shape)

    Y = ad.read_h5ad(par["target_path"]).to_df()
    num_pro = Y.shape[1]
    if args.sub:
        var = var.iloc[overlapped_idx]
        Y = Y.iloc[:,overlapped_idx]
    labels = torch.from_numpy(Y.to_numpy())
    print(labels.shape)

elif subtask == 'gex2adt':
    par = {
        "input_path": osp.join(config.PROCESSED_DATA_DIR, 'svd', f'train_{subtask}_inputs_svd128.h5ad'),
        "test_input_path": osp.join(config.PROCESSED_DATA_DIR, 'svd', f'test_{subtask}_inputs_svd128.h5ad'),
        "gene_adj_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_aligned_{subtask}.npz'),
        "src_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_inputs.h5ad'),
        "test_src_path": osp.join(config.RAW_DATA_DIR, f'test_{subtask}_inputs.h5ad'),
        "target_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_targets.h5ad'),
        "test_target_path": osp.join(config.RAW_DATA_DIR, f'test_{subtask}_targets.h5ad'),
        "var_path": osp.join(config.RAW_DATA_DIR, f'train_{subtask}_targets.h5ad'),
        "symbols_path": osp.join(config.PROCESSED_DATA_DIR, f'gene_protein_edges_{subtask}.h5'),
        "ppi_path": osp.join(config.PROCESSED_DATA_DIR, f'string_human_ppi_v11.5_target_{subtask}.npz'),
        "output_path": osp.join(config.OUT_DIR,  f'gnn_trans_trans_batch_{subtask}'),
    }
    var = ad.read_h5ad(par["var_path"], backed="r").var
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
    overlapped_idx = sorted(set(list(ppi['head']) + list(ppi['tail'])))
    X_src = ad.read_h5ad(par["src_path"]).to_df()
    X_src_test = ad.read_h5ad(par["test_src_path"]).to_df()
    X_src_test.columns = X_src.columns
    obs = torch.from_numpy(pd.concat([X_src, X_src_test]).to_numpy())
    print(obs.shape)
    
    Y = ad.read_h5ad(par["target_path"]).to_df()
    Y_test = ad.read_h5ad(par["test_target_path"]).to_df()
    num_pro = Y.shape[1]
    if args.sub:
        var = var.iloc[overlapped_idx]
        Y = Y.iloc[:,overlapped_idx]
        Y_test = Y_test.iloc[:,overlapped_idx]
    labels = torch.from_numpy(pd.concat([Y, Y_test]).to_numpy())
    print(labels.shape)

if subtask == 'cite':
    act = 'selu'
    sage_agg = 'mean'
    pretrain_epoch = 100
    pre_tol = 40
    feat_drop = 0.5
    attn_drop = 0.1
    training_edge = False
    pos_enc_dim = 10
    layers = [512, 512, 512, 512]
    # layers = [512, 512, 512]
    batch_size = 3000
    patience = 20
    tol = 40
    weight_decay = 1e-6
    # weight_decay = 1e-5
    lr = 1e-3
    gat_head = 4
    trans_head = 8
    local_gnn_type = 'SAGE'
    global_model_type = 'Performer'
    pna_degrees = None

elif subtask == 'gex2adt':
    act = 'selu'
    sage_agg = 'mean'
    pretrain_epoch = 100
    pre_tol = 40
    feat_drop = 0.5
    # feat_drop = 0.6
    attn_drop = 0.1
    # attn_drop = 0.3
    training_edge = False
    pos_enc_dim = 10
    # pos_enc_dim = 20
    layers = [512, 512, 512, 512]
    # layers = [1024, 1024, 1024, 1024]
    # layers = [512, 512, 512]
    batch_size = 10000
    patience = 20
    tol = 40
    weight_decay = 1e-5
    # weight_decay = 5e-6
    lr = 1e-3
    gat_head = 4
    trans_head = 8
    local_gnn_type = 'SAGE'
    global_model_type = 'Performer'
    pna_degrees = None
logging.info(f"{layers}, {lr}, {weight_decay}, {patience}, {tol}, \
    {gat_head}, {trans_head}, {local_gnn_type}, \
    {global_model_type}, {pna_degrees}, {feat_drop}, {attn_drop}")

corrs = []
mses = []
maes = []
for j, sd in enumerate(args.seeds):
    set_seed(sd)
    logging.info(f'Try {j+1}, Seed {sd}')
    if subtask == 'cite':
        train_idx, valid_idx = train_test_split(train_idx, train_size=0.8, random_state=sd)
    elif subtask == 'gex2adt':
        train_idx, valid_idx = train_test_split(train_idx, train_size=0.85, random_state=sd)
    num_cell = obs.shape[0]
    num_gene = obs.shape[1]
    gene_feat = torch.mm(obs.T, feat)

    if args.sub:
        num_pro_sub = len(overlapped_idx)
    else: 
        num_pro_sub = num_pro
        overlapped_idx = None
    
    pro_feat = None
    g0 = create_graph(num_cell, num_gene, num_pro, obs, symbols, ppi, 
        feat, labels, gene_adj, gene_feat, pro_feat, training_edge=training_edge, pro_sub_idx=overlapped_idx)
    
    if args.init_pro:
        g0.nodes['protein'].data['feat'] = torch.zeros(num_pro_sub, feat_dim).float()
        g_gene_pro = dgl.to_homogeneous(g0.edge_type_subgraph(['ppi', 'coexp', 'symbol']), ndata=['feat'], edata=['edge_weight'])
        appnp = dglnn.APPNPConv(5, 0.5)
        feat_pro = appnp(g_gene_pro, g_gene_pro.ndata['feat'], g_gene_pro.edata['edge_weight'])[-num_pro_sub:]
        g0.nodes['protein'].data['feat'] = feat_pro.cpu().float()

    g_pro = dgl.to_homogeneous(g0.edge_type_subgraph(['ppi']), ndata=['feat'], edata=['edge_weight'])
    g_gene = dgl.to_homogeneous(g0.edge_type_subgraph(['coexp']), edata=['edge_weight'])    
    if args.pretrain:
        graph_edata = (g_pro.edata['edge_weight'], g_pro.edges())
        shape = (g_pro.num_nodes(), g_pro.num_nodes())
        ppi_adj = torch.tensor(coo_matrix(graph_edata, shape).todense())

        logging.info("Pretraining")
        mse = nn.MSELoss()
        model = GAE([feat_dim, feat_dim, feat_dim], g_pro.num_nodes()).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        pre_mses = []
        counter = 0
        for i in tqdm(range(pretrain_epoch)):
            feat_pro, ppi_recon = model(g_pro.to(device))
            recon_loss = mse(ppi_recon, ppi_adj.to(device))
            opt.zero_grad()
            recon_loss.backward()
            opt.step()
            logging.info(f'pretrain epoch {i + 1}, recon loss: {recon_loss.cpu().item():0.8f}')
            pre_mses.append(recon_loss.cpu())
            if recon_loss.cpu() > min(pre_mses):
                counter += 1
            if counter == pre_tol:
                logging.info("Pretraining finished")
                break
                
        g0.nodes['protein'].data['feat'] = feat_pro.detach().cpu().float()

    pos_enc = args.pos_enc
    if pos_enc == 'lap':
        if args.pro_mod in ['gps', 'trans'] :
            pro_lap_pos_enc = laplacian_positional_encoding(g_pro, pos_enc_dim)
            g0.nodes['protein'].data['pos_enc'] = pro_lap_pos_enc
        
        if args.gene_mod in ['gps', 'trans'] :
            gene_lap_pos_enc = laplacian_positional_encoding(g_gene, pos_enc_dim)
            g0.nodes['gene'].data['pos_enc'] = gene_lap_pos_enc

    elif pos_enc == 'wl':
        if args.pro_mod in ['gps', 'trans'] :
            pro_wl_pos_enc = wl_positional_encoding(g_pro)
            g0.nodes['protein'].data['pos_enc'] = pro_wl_pos_enc
        
        if args.gene_mod in ['gps', 'trans'] :
            gene_wl_pos_enc = wl_positional_encoding(g_gene)
            g0.nodes['gene'].data['pos_enc'] = gene_wl_pos_enc

    elif pos_enc == 'rw':
        if args.pro_mod in ['gps', 'trans'] :
            pro_rw_pos_enc = dgl.random_walk_pe(g_pro, pos_enc_dim, 'edge_weight')
            g0.nodes['protein'].data['pos_enc'] = pro_rw_pos_enc
        if args.gene_mod in ['gps', 'trans'] :
            gene_rw_pos_enc = dgl.random_walk_pe(g_gene, pos_enc_dim, 'edge_weight')
            g0.nodes['gene'].data['pos_enc'] = gene_rw_pos_enc.cpu()
    
    train_loader = DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_idx, batch_size=batch_size, shuffle=True)
    model = ScMoFormer(dims=[num_gene] + layers + [num_pro_sub], feat_dim=feat_dim, pos_enc=pos_enc,
        num_pro=num_pro_sub, num_cell=num_cell, num_gene=num_gene, pos_enc_dim=pos_enc_dim, local_gnn_type=local_gnn_type, 
        global_model_type=global_model_type, gat_head=gat_head, trans_head=trans_head, cell_mod=args.cell_mod,
        pna_degrees=pna_degrees, gene_mod=args.gene_mod, pro_mod=args.pro_mod, agg=args.agg,
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
    # out_path = f'gnn_trans_trans_batch_{subtask}_{args.gene_mod}_{args.pro_mod}_{args.cell_mod}_{args.agg}_{args.pos_enc}_{sd}'
    out_path = f'gnn_trans_trans_batch_{subtask}_{args.gene_mod}_{args.pro_mod}_{args.cell_mod}_{args.agg}_{args.pos_enc}_{sd}_{args.run_id}'
    graphs = {}
    for k, batch in enumerate(train_loader):
        node_dict = {
            'cell': batch.int(),
            'protein': torch.arange(num_pro_sub).int(), 
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
                logging.info(f'early stopping counter: {counter}')
            else:
                torch.save(model.state_dict(), f'saved_models/{out_path}.pth')
        if counter == tol:
            logging.info(f'Early stopped. Best val corr: {-min(tempcorrs):0.8f}, \
                best val MSE: {min(tempmses):0.8f}, best val MAE: {min(tempmaes):0.8f}')
            # torch.save(model.state_dict(), f'saved_models/{out_path}.pth')
            break
        scheduler.step(val)
        logging.info(f'epoch {i + 1}, training: {loss.cpu().item():0.8f}, \
            val corr: {valcorr:0.8f}, val MSE: {valmse:0.8f}, val MAE: {valmae:0.8f} \
            test corr: {testcorr:0.8f}, test MSE: {testmse:0.8f}, test MAE: {testmae:0.8f}')
    
    model.load_state_dict(torch.load(f'saved_models/{out_path}.pth'))
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
    logging.info(f"Try {j + 1}, corr = {testcorr:.8f}, MSE = {testmse:.8f}, MAE = {testmae:.8f}")
    
    # logging.info("Generate anndata object ...")
    # adata_pred = ad.AnnData(X=csr_matrix(Z), obs=meta_test, var=var, dtype=np.float32)
    # logging.info('Storing annotated data...')
    # adata_pred.write_h5ad(osp.join(config.OUT_DIR, subtask, f"{out_path}.h5ad"), compression="gzip")
    

rmses = np.sqrt(mses)
logging.info(f'Corr: {np.mean(corrs):.5f} +/- {np.std(corrs):.5f}')
logging.info(f'MSE: {np.mean(mses):.5f} +/- {np.std(mses):.5f}')
logging.info(f'RMSE: {np.mean(rmses):.5f} +/- {np.std(rmses):.5f}')
logging.info(f'MAE: {np.mean(maes):.5f} +/- {np.std(maes):.5f}')
logging.info(f'Corrs: {corrs}')
logging.info(f'MSEs: {mses}')
logging.info(f'RMSEs: {rmses}')
logging.info(f'MAEs: {maes}')