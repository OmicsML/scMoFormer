import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import Performer

from layers import GPSLayer


class ScMoFormer(nn.Module):
    def __init__(self, dims, feat_dim, num_pro, pos_enc, pos_enc_dim=10, local_gnn_type='SAGE', 
        global_model_type='Performer', trans_head=8, pna_degrees=None, feat_drop=0.1, attn_drop=0.1, 
        act='selu', sage_agg='mean'):
        super().__init__()
        self.layers = len(dims) - 1
        self.pos_enc = pos_enc

        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'selu':
            self.act = nn.SELU()
        self.dp = nn.Dropout(feat_drop)

        self.sageconvs = nn.ModuleList()
        self.pro_gps = nn.ModuleList()
        self.gene_gps = nn.ModuleList()
        self.cell_trans = nn.ModuleList()
        self.mlpcell = nn.ModuleList()

        self.edges = ['obs', 'rev_obs', 'coexp', 'ppi', 'symbol', 'rev_symbol']
        self.edges_sub = ['obs', 'rev_obs', 'symbol', 'rev_symbol']
        self.nodes = ['gene', 'protein', 'cell']
        for i in range(1, self.layers-1):    
            self.sageconvs.append(
                dglnn.HeteroGraphConv(
                    dict(zip(
                            self.edges, [
                                dglnn.SAGEConv(
                                    dims[i],
                                    dims[i + 1],
                                    aggregator_type=sage_agg,
                                    feat_drop=feat_drop,
                                ) 
                                for j in range(len(self.edges))
                            ]
                    )),
                    aggregate='stack'
                )
            )
            self.pro_gps.append(
                GPSLayer(dims[i], local_gnn_type, global_model_type, trans_head, self.act, pna_degrees, 
                         dropout=feat_drop, attn_dropout=attn_drop))
            self.gene_gps.append(
                GPSLayer(dims[i], local_gnn_type, global_model_type, trans_head, self.act, pna_degrees, 
                         dropout=feat_drop, attn_dropout=attn_drop))
            self.cell_trans.append(
                Performer(dims[i], depth=1, heads=trans_head, dim_head=int(dims[i] / trans_head), attn_dropout=attn_drop,
                          ff_dropout=feat_drop, ff_mult=2, kernel_fn=self.act, causal=False))
            self.mlpcell.append(nn.Linear(dims[i], dims[i + 1]))
            
        self.emb_cell = nn.Linear(feat_dim, dims[1])
        self.emb_gene = nn.Linear(feat_dim, dims[1])
        self.emb_protein = nn.Embedding(num_pro, dims[1])
        self.emb_gene_pos_enc = nn.Linear(pos_enc_dim, dims[1])
        self.emb_pro_pos_enc = nn.Linear(pos_enc_dim, dims[1])

        self.mlp_last = nn.Linear(sum(dims[1:-1]), dims[-1])
        self.conv_norm1 = nn.ModuleList()
        self.conv_norm2 = nn.ModuleList()
        for i in range((self.layers-2) * len(self.nodes)):
            self.conv_norm1.append(nn.LayerNorm(dims[i // len(self.nodes) + 2]))
            self.conv_norm2.append(nn.LayerNorm(dims[i // len(self.nodes) + 2]))

    def conv(self, graph, layer, h):
        hmlp_cell = self.conv_norm1[layer * len(self.nodes) + 2](self.act(self.mlpcell[layer](h['cell'])))
        
        g_gene = dgl.to_homogeneous(graph.edge_type_subgraph(['coexp']), edata=['edge_weight'])
        g_pro = dgl.to_homogeneous(graph.edge_type_subgraph(['ppi']), edata=['edge_weight'])
        g_sub = graph.edge_type_subgraph(self.edges_sub)
        hsage = self.sageconvs[layer](g_sub, h, mod_kwargs=dict(zip(self.edges, [{'edge_weight':
            self.dp(graph.edges['obs'].data['edge_weight'])}] + [{'edge_weight': 
            self.dp(graph.edges['rev_obs'].data['edge_weight'])}] + [{'edge_weight':
            graph.edges[self.edges_sub[i]].data['edge_weight']} for i in range(2, len(self.edges_sub))])))


        hpro_pro = self.pro_gps[layer](g_pro, h['protein'], graph.edges['ppi'].data['edge_weight'])
        hgene_gene = self.gene_gps[layer](g_gene, h['gene'], graph.edges['coexp'].data['edge_weight'])
        hcell_cell = self.cell_trans[layer](h['cell'].unsqueeze(0))[-1,:,:]
        hcell_cell = self.conv_norm1[layer * len(self.nodes) + 2](self.act(hcell_cell))
        
        hgene_gene = self.conv_norm1[layer * len(self.nodes)](self.act(hgene_gene))
        hpro_pro = self.conv_norm1[layer * len(self.nodes) + 1](self.act(hpro_pro))
        hsage['gene'] = self.conv_norm1[layer * len(self.nodes)](self.act(hsage['gene']))
        hsage['protein'] = self.conv_norm1[layer * len(self.nodes) + 1](self.act(hsage['protein']))
        hsage['cell'] = self.conv_norm1[layer * len(self.nodes) + 2](self.act(hsage['cell']))

        hpro = hsage['protein'].sum(1) + hpro_pro
        hgen = hsage['gene'].sum(1) + hgene_gene
        hcel = hsage['cell'].sum(1) + hmlp_cell + hcell_cell
        
        hgene = self.conv_norm2[layer * len(self.nodes)](self.act(hgen))
        hprotein = self.conv_norm2[layer * len(self.nodes) + 1](self.act(hpro))
        hcell = self.conv_norm2[layer * len(self.nodes) + 2](self.act(hcel))
        return {'gene': hgene, 'protein': hprotein, 'cell': hcell}

    def forward(self, graph):
        hpro = self.emb_protein(graph.nodes['protein'].data['feat'])
        hpro += self.emb_pro_pos_enc(graph.nodes['protein'].data['pos_enc'])
        hgene = self.emb_gene(graph.nodes['gene'].data['feat'])
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