# Multi-hop-QA

Llama trained for Multi-Hop Question Answering using  based Supervised Fine Tuning (check llama_new branch)
This final project has been done by,
Lakshay Piplani ljp5718@psu.edu
Durva Dev dbd5616@psu.edu

## Clone the repository
```
git clone https://github.com/LakshayPiplani/multi-hop-QA.git
```

## Dataset

Download the HOTPOT QA dataset from the official URLS

```
cd ~/data/scripts
python download_data.py --output_dir <output_directory> --splits <split1> <split2> <split3>
```

You can specify the output directory and the splits you want to download using command line arguments. You can download train, dev and test splits.

By default, it will download the train and dev splits and save them in the "data/raw" directory. 

## Train Models

You can change base model in train_sft.py

```
cd ~/src
python3 train_sft.py
```

## Inference 

```
cd ~/src
python3 infer_and_evaluate.py
```