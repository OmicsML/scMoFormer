"""Convert raw h5 data to h5ad by combining raw data with metadata."""
import os.path as osp
from glob import glob
import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix

import config


def main(log_level="INFO"):
    logger = config.get_logger("data_to_h5ad", level=log_level)

    metadata_df = pd.read_csv(osp.join(config.RAW_DATA_DIR, "metadata.csv"), index_col=0)
    logger.info(f"metadata:\n{metadata_df}")

    for path in sorted(glob(osp.join(config.RAW_DATA_DIR, "*.h5"))):
        name = osp.splitext(osp.split(path)[1])[0]
        if "day" in name or "multi" in name:
            continue
        logger.info(f"Loading data {name} from {path}")
        df = pd.read_hdf(path)
        logger.info(f"\n{df}")

        adata_x = csr_matrix(df.values)
        obs = metadata_df.align(pd.DataFrame(index=df.index), join="right", axis=0)[0]
        var = pd.DataFrame(index=df.columns)

        adata = ad.AnnData(X=adata_x, obs=obs, var=var)
        logger.debug(f"adata=\n{adata}")
        logger.debug(f"adata.obs=\n{adata.obs}")
        logger.debug(f"adata.var=\n{adata.var}")

        outpath = osp.join(config.RAW_DATA_DIR, f"{name}.h5ad")
        logger.info(f"Saving processed data to {outpath}")
        adata.write_h5ad(outpath)


if __name__ == "__main__":
    main()
