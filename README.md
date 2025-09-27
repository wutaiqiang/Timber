
Code for the paper `Timber: Training-free Instruct Model Refining with Base via Effective Rank`


# Environment

Please follow the official guidance of [Opencompass](https://github.com/open-compass/opencompass?tab=readme-ov-file#-environment-setup) to set up a python environment.

We use the lmdeploy backend, please remember to set
```
pip install "opencompass[lmdeploy]"
```

# Timber

Download the official weights from huggingface:

- [Llama](https://huggingface.co/meta-llama)
- [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)

We recommend to download via the huggingface-cli, such as

```
hf download Qwen/Qwen3-30B-A3B --token $your_hf_token --local-dir weights/Qwen3-30B-A3B/Qwen3-30B-A3B
```

Then, run the timber.py:

```
python3 -u timber.py $path/to/base $path/to/instruct $path/to/save --gamma  0.0 --svd_cache_path $your_path/cache.pt
```

where `gamma` is the scale factor and `svd_cache_path` is the cache file for eRank.

# Evaluate

We employ the opencompass for evaluation.

You need to modify the config files first.

For example,  in `Evaluation/llama_1B.py`, replace the `paths` with your folder, modify the `tp` and `num_gpus` to fit your machine.

Then all you need is to run `opencompass Evaluation/llama_1B.py` and wait the final results.

