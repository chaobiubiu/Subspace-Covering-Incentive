# Subspace Covering Multi-Agent Intrinsic Control

Code for the paper "Subspace Covering Multi-Agent Intrinsic Control" submitted to Neural Networks.

This repository develops SCI algorithm in the SMAC task. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the approach in the paper, run this command:

```train 
python scripts/train_sparse_smac_scripts/train_smac_{map}.py
```

where map can be set to `3m, 2m_vs_1z, 3s_vs_5z, 8m_vs_9m, MMM, MMM2, 2s_vs_1sc, 5m_vs_6m, 3s5z`

## Hyper-parameters

To modify the hyper-parameters of algorithms, refer to:

```
config.py
```

## Note

This repository is developed based on MAPPO. And we have cited it in our work.