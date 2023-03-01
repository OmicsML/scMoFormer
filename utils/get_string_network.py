import os.path as osp
from io import BytesIO
from typing import Dict
import argparse
import anndata as ad
import mygene
import numpy as np
import pandas as pd
import requests

import config


STRING_HUMAN_NETWORK_URL = "https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
STRING_HUMAN_NETWORK_INFO_URL = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"
STRING_HUMAN_NETWORK_ALIAS_URL = "https://stringdb-static.org/download/protein.aliases.v11.5/9606.protein.aliases.v11.5.txt.gz"
OUT_FILE_NAME = "string_human_ppi_v11.5"
SYMBOL_MAPPING_FILE = "symbol_mapping.csv"
TARGET_PPI_FILE = "string_human_ppi_v11.5_target"
GENE_SYMBOL_FILE = "gene_protein_edges"

logger = config.get_logger("string", level="INFO")

def _download():
    logger.info(f"Downloading data from STRING")
    r = requests.get(STRING_HUMAN_NETWORK_URL)
    if not r.ok:
        raise requests.exceptions.RequestException(f"Failed: {r!r}")
    edge_df = pd.read_csv(BytesIO(r.content), compression="gzip", sep=" ")
    edge_df.combined_score /= 1000  # renormalize back to probabilties

    r_info = requests.get(STRING_HUMAN_NETWORK_INFO_URL)
    if not r_info.ok:
        raise requests.exceptions.RequestException(f"Failed: {r_info!r}")
    info_df = pd.read_csv(BytesIO(r_info.content), compression="gzip", sep="\t")

    """
    r_alias = requests.get(STRING_HUMAN_NETWORK_ALIAS_URL)
    if not r_alias.ok:
        raise requests.exceptions.RequestException(f"Failed: {r_alias!r}")
    alias_df = pd.read_csv(BytesIO(r_alias.content), compression="gzip", sep="\t")
    """

    logger.info(f"STRING network downloaed:\n{edge_df}")
    return edge_df, info_df


def _get_gene_id_map():
    """Obtain mapping from gene ensembl to gene id from data.

    The gene id is constructed by joining the gene ensembl and gene symbol
    using "_".

    Note:
        If gene symbol not available from the original data, then use gene
        ensemble alone as the gene id.

    Example:
        - gene_ensembl: ENSG00000121410, gene_symbol: A1BG, gene_id:
        ENSG00000121410_A1BG
        - gene_ensembl: ENSG00000203995, gene_symbol: (N/A), gene_id:
        ENSG00000203995

    """
    gene_df = None
    file_names = ["train_cite_inputs.h5ad", "test_cite_inputs.h5ad", "train_multi_targets.h5ad"]
    for file_name in file_names:
        path = osp.join(config.RAW_DATA_DIR, file_name)
        if not osp.isfile(path):
            raise FileNotFoundError(f"{path} not exist, run utils/data_to_h5ad.py first")

        logger.info(f"Loading gene list from {path}")
        new_df = pd.DataFrame(ad.read_h5ad(path, backed="r").var.index.tolist())[0].str.split("_", expand=True)
        gene_df = new_df if gene_df is None else gene_df.merge(new_df, how="outer")
        logger.info(f"\tCurrent number of genes: {gene_df.shape[0]:,}")

    gene_df.columns = ["gene_ensembl", "gene_symbol"]
    logger.info(f"Total number of genes from data = {gene_df.gene_ensembl.unique().size:,}\n{gene_df}")

    gene_id_map = {}  # gene_ensembl -> gene_id
    for _, (ens, sym) in gene_df.iterrows():
        gene_id_map[ens] = "_".join((ens, sym)) if isinstance(sym, str) else ens

    return gene_id_map


def _convert_and_save(edge_df, info_df, gene_ids, pro_ids, subtask, gene_id_map=None):
    # Align gene interaction network node ids with data
    edge_df.protein1 = edge_df.protein1.apply(lambda x: x.split(".")[1])
    edge_df.protein2 = edge_df.protein2.apply(lambda x: x.split(".")[1])
    ensps = sorted(set(edge_df.protein1.tolist() + edge_df.protein2.tolist()))
    network_genes_df = mygene.MyGeneInfo().querymany(ensps, scopes="ensembl.protein", fields="ensembl.gene",
                                                     species="human", as_dataframe=1)
    network_genes_df = network_genes_df.reset_index()[["query", "ensembl.gene"]].rename(columns={
        "query": "protein_ensembl",
        "ensembl.gene": "gene_ensembl"
    })
    if gene_id_map is not None:
        network_genes_df["gene_id"] = network_genes_df.gene_ensembl.apply(gene_id_map.get)
    else:
        network_genes_df["gene_id"] = network_genes_df["gene_ensembl"]
    
    # Align PPI with target proteins
    info_df["ensp"] = info_df["#string_protein_id"].apply(lambda x: x.split(".")[1])
    ensp_to_sym = {}
    for i in info_df.index:
        ensp_to_sym[info_df["ensp"][i]] = info_df["preferred_name"][i]
    symbol_mapping = pd.read_csv(osp.join(config.PROCESSED_DATA_DIR, SYMBOL_MAPPING_FILE))
    symbol_mapping = symbol_mapping[symbol_mapping["protein"].isin(pro_ids)]
    target_symbols = list(symbol_mapping["symbol"])
    sym_to_pro = {}
    for i in symbol_mapping.index:
        sym_to_pro[symbol_mapping["symbol"][i]] = symbol_mapping["protein"][i]
    pro_to_idx = {j: i for i, j in enumerate(pro_ids)}

    target_pro_ppi = edge_df.copy()
    target_pro_ppi.protein1 = target_pro_ppi.protein1.apply(ensp_to_sym.get)
    target_pro_ppi.protein2 = target_pro_ppi.protein2.apply(ensp_to_sym.get)
    target_pro_ppi = target_pro_ppi[target_pro_ppi.protein1.isin(target_symbols)]
    target_pro_ppi = target_pro_ppi[target_pro_ppi.protein2.isin(target_symbols)]
    target_pro_ppi.protein1 = target_pro_ppi.protein1.apply(sym_to_pro.get).apply(pro_to_idx.get)
    target_pro_ppi.protein2 = target_pro_ppi.protein2.apply(sym_to_pro.get).apply(pro_to_idx.get)

    # Find connections between gene and protein by symbols
    symbol_df = pd.DataFrame(0, index=gene_ids, columns=pro_ids)
    gene_to_idx = {j: i for i, j in enumerate(gene_ids)}
    gene_series = pd.Series(gene_ids)
    gene_symbols = gene_series.apply(lambda x: x.split("_")[1].upper())
    gene_match_idx = [x for x in range(len(gene_series)) if str(gene_symbols.iloc[x]) in target_symbols]
    for i in gene_match_idx:
        if sym_to_pro[gene_symbols[i]] in symbol_df.columns:
            symbol_df.loc[symbol_df.index[i], sym_to_pro[gene_symbols[i]]] = 1
        
    # Try to map network protein to gene id > gene ensembl > protein.
    ensp_to_gid = {}  # ensp -> gene id / ensg / ensp
    for _, row in network_genes_df.iterrows():
        ensp = row["protein_ensembl"]
        if isinstance(gid := row["gene_id"], str):
            ensp_to_gid[ensp] = gid
        elif isinstance(ensg := row["gene_ensembl"], str):
            ensp_to_gid[ensp] = ensg
        else:
            ensp_to_gid[ensp] = ensp
    
    ensp_to_ensg = {}  # ensp -> ensg / ensp
    for _, row in network_genes_df.iterrows():
        ensp = row["protein_ensembl"]
        if isinstance(ensg := row["gene_ensembl"], str):
            ensp_to_ensg[ensp] = ensg
        else:
            ensp_to_ensg[ensp] = ensp
    ensp_to_idx = {j: i for i, j in enumerate(ensps)}  # ensp -> index

    converted_edge_df = edge_df.copy()
    converted_edge_df.protein1 = converted_edge_df.protein1.apply(ensp_to_gid.get)
    converted_edge_df.protein2 = converted_edge_df.protein2.apply(ensp_to_gid.get)

    head = edge_df.protein1.apply(ensp_to_idx.get).values
    tail = edge_df.protein2.apply(ensp_to_idx.get).values
    weight = edge_df.combined_score.values.astype(np.float32)
    ids = list(map(ensp_to_gid.get, ensps))
    ensgs = list(map(ensp_to_ensg.get, ensps))
    target_head = target_pro_ppi.protein1.values
    target_tail = target_pro_ppi.protein2.values
    target_weight = target_pro_ppi.combined_score.values.astype(np.float32)

    out_path = osp.join(config.PROCESSED_DATA_DIR, f"{OUT_FILE_NAME}.npz")
    np.savez_compressed(out_path, head=head, tail=tail, weight=weight, ids=ids, ensps=ensps, ensgs=ensgs)
    target_path = osp.join(config.PROCESSED_DATA_DIR, f"{TARGET_PPI_FILE}_{subtask}.npz")
    np.savez_compressed(target_path, head=target_head, tail=target_tail, weight=target_weight, ensps=ensps)
    symbol_df.to_hdf(osp.join(config.PROCESSED_DATA_DIR, f"{GENE_SYMBOL_FILE}_{subtask}.h5"), "df")
    logger.info(f"Processed files saved to {config.PROCESSED_DATA_DIR}")
    return converted_edge_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", type=str, default="cite") # "cite", "gex2adt"
    args = parser.parse_args()
    subtask = args.subtask
    gene_info = ad.read_h5ad(osp.join(config.RAW_DATA_DIR, f"train_{subtask}_inputs.h5ad"), backed="r")
    pro_info = ad.read_h5ad(osp.join(config.RAW_DATA_DIR, f"train_{subtask}_targets.h5ad"), backed="r")
    if subtask == "cite":
        gene_ids = gene_info.var.index
        gene_id_map = _get_gene_id_map()
    elif subtask == "gex2adt":
        gene_ensgs = gene_info.var["gene_ids"]
        gene_symbols = [str(x).upper() for x in gene_info.var.index]
        gene_ids = [f"{gene_ensgs[x]}_{gene_symbols[x]}" for x in range(len(gene_ensgs))]
        gene_id_map = None
    pro_ids = [str(x).upper() for x in pro_info.var.index]
    edge_df, info_df = _download()
    _convert_and_save(edge_df, info_df, gene_ids, pro_ids, subtask, gene_id_map)



if __name__ == "__main__":
    main()
