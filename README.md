# sac-lili

Files:
- `softlearning/algorithms/sac.py`: implements SAC and latent model
- `softlearning/replay_pools/multitask_replay_pool.py`: implements replay pool separated by task
- `softlearning/samplers/simple_sampler.py`: encodes trajectory from previous task and samples from the environment with policy conditioned on encoding

Setup:

```
cd softlearning
conda env create -f environment.yml
conda activate sac_lili
pip install -e .
```

The hyperparameters for LILI are in `examples/development/variants.py`.