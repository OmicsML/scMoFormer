import argparse
import math
import os.path as osp
from pickle import FALSE
from time import time

import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

import config

logger = config.get_logger("svd")
t1 = time()

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-t', '--subtask', default='cite', type=str)
parser.add_argument('-n', '--n_comp', default=128, type=int)
args = parser.parse_args()
n_comp = args.n_comp
subtask = args.subtask

config.ensure_dir(osp.join(config.PROCESSED_DATA_DIR, 'svd'))
par = {
    'input_train_mod1': osp.join(config.RAW_DATA_DIR, f'train_{subtask}_inputs.h5ad'),
    'input_test_mod1': osp.join(config.RAW_DATA_DIR, f'test_{subtask}_inputs.h5ad'),
    'train_output': osp.join(config.PROCESSED_DATA_DIR, 'svd', f'train_{subtask}_inputs_svd{n_comp}.h5ad'),
    'test_output': osp.join(config.PROCESSED_DATA_DIR, 'svd', f'test_{subtask}_inputs_svd{n_comp}.h5ad'),
}  

X_train = ad.read_h5ad(par['input_train_mod1'])
X_test = ad.read_h5ad(par['input_test_mod1'])
obs_train = X_train.obs
obs_test = X_test.obs
X_train = X_train.to_df()
X_test = X_test.to_df()

n_train = len(X_train)
X_train = X_train.loc[:,X_test.columns]
df = pd.concat([X_train, X_test])

logger.info('Running truncated SVD ...')
df = df.to_numpy()
data_sparse = csr_matrix(df)
logger.info(f'data shape: {data_sparse.shape}')
embedder = TruncatedSVD(n_components=n_comp, random_state=10)
embeddings = embedder.fit_transform(data_sparse)
logger.info(embedder.explained_variance_ratio_)
logger.info(f"Total expressed var: {np.sum(embedder.explained_variance_ratio_)}")
X_train = embeddings[:n_train,:]
X_test = embeddings[n_train:,:]

logger.info("Generate anndata object ...")
logger.info(f'X_train shape: {X_train.shape}')
logger.info(f'X_test shape: {X_test.shape}')
X_train = csr_matrix(X_train)
X_test = csr_matrix(X_test)
adata_train = ad.AnnData(X=X_train, obs=obs_train)
adata_test = ad.AnnData(X=X_test, obs=obs_test)

logger.info('Storing data...')
adata_train.write_h5ad(par['train_output'], compression="gzip")
adata_test.write_h5ad(par['test_output'], compression="gzip")
t2 = time()
logger.info('Running time: %d seconds' % math.ceil(t2 - t1))
