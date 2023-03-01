import torch
import dgl
from scipy.sparse import coo_matrix

def create_graph(num_cell, num_gene, num_pro, X_prime, symbols, ppi, cell_feat, Y, 
                 gene_adj=None, gene_feat=None, pro_feat=None):
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