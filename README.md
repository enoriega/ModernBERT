# Welcome!

This is the repository where you can find ModernBERT, our experiments to bring BERT into modernity via both architecture changes and scaling.

This repository noticeably introduces FlexBERT, our modular approach to encoder building blocks, and heavily relies on .yaml configuration files to build models. The codebase builds upon [MosaicBERT](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert), and specifically the [unmerged fork bringing Flash Attention 2](https://github.com/Skylion007/mosaicml-examples/tree/skylion007/add-fa2-to-bert) to it, under the terms of its Apache 2.0 license. We extend our thanks to MosaicML for starting the work on modernising encoders!

This README is very barebones and is still under construction. It will improve with more reproducibility and documentation in the new year, as we gear up for more encoder niceties after the pre-holidays release of ModernBERT. For now, we're mostly looking forward to seeing what people build with the [🤗 model checkpoints](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb)).

For more details on what this repository brings, we recommend reading our [release blog post](https://huggingface.co/blog/modernbert) for a high-level overview, and our [arXiv preprint](https://arxiv.org/abs/2412.13663) for more technical details.

All code used in this repository is the code used as part of our experiments for both pre-training and GLUE evaluations, there's no uncommitted secret training sauce.

**This is the research repository for ModernBERT, focused on pre-training and evaluations. If you're seeking the HuggingFace version, designed to integrate with any common pipeline, please head to the [ModernBERT Collection on HuggingFace](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb)**

*ModernBERT is a collaboration between [Answer.AI](https://answer.ai), [LightOn](https://lighton.ai), and friends.*

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.12). Install uv, then create the environment:

```bash
# CPU-only / local dev (no Flash Attention): core deps only
uv sync

# GPU training: add Flash Attention 2 (prebuilt wheel, Linux + CUDA)
uv sync --extra flash

# optional extras: data prep tooling and ColBERT retrieval evals
uv sync --extra flash --extra data --extra colbert
```

Run any entry point through the environment with `uv run`, e.g. `uv run composer main.py <config.yaml>`.

Notes on Flash Attention:
- Flash Attention is **optional**. Without it, FlexBERT automatically falls back to PyTorch's SDPA attention, so `uv sync` alone is enough for development and CPU runs.
- The `flash` extra installs a prebuilt `flash-attn==2.8.3` wheel for cp312 / torch 2.7 / CUDA 12 (cxx11 ABI TRUE). A cu12 wheel runs fine on CUDA 13 hosts — torch and flash-attn bundle their own CUDA runtime, and the NVIDIA driver is backward compatible.
- H100 Flash Attention 3 is still built separately:
  ```bash
  # git clone https://github.com/Dao-AILab/flash-attention.git
  # cd flash-attention/hopper && python setup.py install
  ```

## Training

Training heavily leverages the [composer](https://github.com/mosaicml/composer) framework. All training are configured via YAML files, of which you can find examples in the `yamls` folder. We highly encourage you to check out one of the example yamls, such as `yamls/main/flex-bert-rope-base.yaml`, to explore the configuration options.

### Launch command example
To run a training job using `yamls/main/modernbert-base.yaml` on all available GPUs, use the following command.
```
uv run composer main.py yamls/main/modernbert-base.yaml
```

### Data

There are two dataset classes to choose between:

`StreamingTextDataset`
* inherits from [StreamingDataset](https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/dataset_format.html)
* uses MDS, CSV/TSV or JSONL format
* Supports both text and tokenized data
* can be used with local data as well
* WARNING: we found distribution of memory over accelerators to be uneven

`NoStreamingDataset`
* requires decompressed MDS-format, compressed MDS-data can be decompressed using [src/data/mds_conversion.py](src/data/mds_conversion.py)  with the `--decompress` flag.
* Supports both text and tokenized data

When data is being accessed from local, we recommend using `NoStreamingDataset` as it enabled higher training throughput in our setting. Both classes are located in [src/text_data.py](src/text_data.py), and the class to be used for a dataset can be set for each data_loader and dataset by setting streaming: true (StreamingTextDataset) or false (NoStreamingDataset).

```
train_loader:
  name: text
  dataset:
    streaming: false
```

To get started, you can experiment with c4 data using the [following instructions](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert#prepare-your-data).


## Evaluations

### GLUE

GLUE evaluations for a ModernBERT model trained with this repository can be ran with via `run_evals.py`, by providing it with a checkpoint and a training config. To evaluate non-ModernBERT models, you should use `glue.py` in conjunction with a slightly different training YAML, of which you can find examples in the `yamls/finetuning` folder.

### Retrieval

The `examples` subfolder contains scripts for training retrieval models, both dense models based on [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and ColBERT models via the [PyLate](https://github.com/lightonai/pylate) library:
- `examples/train_pylate.py`: The boilerplate code to train a ModernBERT-based ColBERT model with PyLate.
- `examples/train_st.py`: The boilerplate code to train a ModernBERT-based dense retrieval model with Sentence Transformers.
- `examples/evaluate_pylate.py`: The boilerplate code to evaluate a ModernBERT-based ColBERT model with PyLate.
- `examples/evaluate_st.py`: The boilerplate code to evaluate a ModernBERT-based dense retrieval model with Sentence Transformers.


## Reference

If you use ModernBERT in your work, be it the released models, the intermediate checkpoints (release pending) or this training repository, please cite:

```bibtex
@misc{modernbert,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference}, 
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663}, 
}
```