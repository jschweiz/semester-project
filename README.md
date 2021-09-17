# softlearning-ctrl

Running locally:
`softlearning run_example_local examples.development --universe=gym --domain=Point2DEnv --task=Default-v0
--exp-name=exp-name --checkpoint-frequency=200`

Files:
- `softlearning/algorithms/sac.py`: implements SAC and latent model
- `softlearning/replay_pools/multitask_replay_pool.py`: implements replay pool separated by task
- `softlearning/samplers/simple_sampler.py`: encodes trajectory from previous task and samples from the environment with policy conditioned on encoding

Setup:

```
cd softlearning-ctrl
conda env create -f environment.yml
conda activate softlearning
pip install -e .
```
