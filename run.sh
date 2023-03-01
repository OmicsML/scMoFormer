#!/bin/bash

# cite, gex2adt
echo "Subask: cite or gex2adt"
read TASK

echo "Download data and preprocess? y or n"
read FLAG

if [[ ${FLAG} == "y" ]]
then
    python ./utils/download_data.py --subtask ${TASK}
    if [[ ${TASK} == "cite" ]]
    then
        python ./utils/data_to_h5ad.py
    fi
    python ./utils/svd.py --subtask ${TASK}
    # python ./utils/get_string_network.py --subtask ${TASK}
fi

python train.py --subtask ${TASK}
echo ${TASK}