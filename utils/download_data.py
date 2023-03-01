from dance.datasets.multimodality import ModalityPredictionDataset
import os
import os.path as osp
import argparse

import config


DATA_DIR = "openproblems_bmmc_cite_phase2_rna"
DATA_FILE = "openproblems_bmmc_cite_phase2_rna.censor_dataset.output"

parser = argparse.ArgumentParser()
parser.add_argument("--subtask", type=str, default="cite")
args = parser.parse_args()
subtask = args.subtask

if subtask == "gex2adt":
    ModalityPredictionDataset(subtask, data_dir=str(config.RAW_DATA_DIR)).download_data()
    splits = ["train", "test"]
    mods = ["mod1", "mod2"]
    outmods = ["inputs", "targets"]
    for split in splits:
        for j in range(len(mods)):
            fpath = osp.join(config.RAW_DATA_DIR, DATA_DIR, f"{DATA_FILE}_{split}_{mods[j]}.h5ad")
            outpath = osp.join(config.RAW_DATA_DIR, f"{split}_{subtask}_{outmods[j]}.h5ad")
            os.system(f"mv {fpath} {outpath}")
elif subtask == "cite":
    os.system("kaggle competitions download -c open-problems-multimodal")
    os.system("unzip open-problems-multimodal.zip -d {config.RAW_DATA_DIR}")
    os.system("rm -f open-problems-multimodal.zip")