import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from performer_pytorch import SelfAttention

class GPSLayer(nn.Module):
    """
        GraphGPS layer 
        adapted from 
        
        "Recipe for a General, Powerful, Scalable Graph Transformer"
        Rampášek et al., 2022
        https://github.com/rampasek/GraphGPS
    """

    def __init__(self, dim_h, local_gnn_type, global_model_type, num_heads, act=nn.ReLU(),
                 pna_degrees=None, dropout=0.0, attn_dropout=0.0, log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.activation = act

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type != 'Transformer':
            raise NotImplementedError(
                "Logging of attention weights is only supported for "
                "Transformer global attention model."
            )

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GIN':
            gin_nn = nn.Sequential(nn.Linear(dim_h, dim_h),
                                   self.activation,
                                   nn.Linear(dim_h, dim_h))
            self.local_model = dglnn.GINConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = dglnn.GATConv(dim_h,
                                             dim_h,
                                             num_heads,
                                             feat_drop=dropout, 
                                             attn_drop=attn_dropout,
                                             residual=True)
        elif local_gnn_type == 'SAGE':
            self.local_model = dglnn.SAGEConv(dim_h,
                                              dim_h,
                                              aggregator_type='mean',
                                              feat_drop=dropout)
        elif local_gnn_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            delta = torch.log1p(torch.from_numpy(np.array(pna_degrees)))
            self.local_model = dglnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             delta=delta,
                                             edge_feat_size=1)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        # Normalization for MPNN and Self-Attention representations.
        self.norm1_local = nn.LayerNorm(dim_h)
        self.norm1_attn = nn.LayerNorm(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.norm2 = nn.LayerNorm(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, graph, h, edge_weight=None):
        h_in1 = h  # for first residual connection
        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            if self.local_gnn_type in ['GIN', 'SAGE', 'PNA']:
                h_local = self.local_model(graph, h, edge_weight)
            else:
                h_local = self.local_model(graph, h)
            if self.local_gnn_type == 'GAT':
                h_local = torch.sum(h_local, 1)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local  # Residual connection.
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h.unsqueeze(0), None, None)[-1,:,:]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h.unsqueeze(0))[-1,:,:]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        h = self.norm2(h)
        return h

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights = True`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s