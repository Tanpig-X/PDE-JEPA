<div align="center">

# PDE-JEPA

### Predictive Representation Learning of Latent Dynamics Modeling for Parametric PDEs

**Zhentao Tan · Jianrong Zhang · Ruijie Quan · Yi Yang**  
Zhejiang University

[![arXiv](https://img.shields.io/badge/arXiv-2609.34715-b31b1b.svg)](https://arxiv.org/abs/2609.34715)
[![Hugging Face Paper](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2609.34715)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://tanpig-x.github.io/PDE-JEPA/)

</div>

**PDE-JEPA learns predictive representations and physics-structured latent dynamics to forecast parametric PDEs from initial states and physical parameters.**

[![PDE-JEPA framework](https://tanpig-x.github.io/PDE-JEPA/assets/method.svg?v=91ee3644ef99)](https://tanpig-x.github.io/PDE-JEPA/)

## TODO

- [x] Project page.
- [x] Training code release.
- [ ] Upload training datasets.

## Results

The [paper](https://arxiv.org/abs/2609.34715) reports the best ID performance on **8 of 9 benchmarks** and the best OOD performance on **all 5 evaluated benchmarks**, with average improvements of **33.4% ID** and **51.4% OOD**.

| Benchmark | ID Relative L2 ↓ | OOD Relative L2 ↓ |
|---|---:|---:|
| Advection | 0.0074 | — |
| Burgers | 0.0428 | — |
| Heat | 0.0274 | — |
| Wave-B | 0.0350 | — |
| Combined | 0.0074 | 0.008 |
| Wave-2D | 0.1140 | 0.157 |
| Vorticity | 0.0348 | 0.288 |
| Heterogeneous Vorticity | 0.0089 | 0.011 (viscosity) / 0.103 (forcing) |
| Gray–Scott | 0.0284 | 0.033 |

Results are from Tables 1 and 4 of the paper. A dash indicates that no OOD result is reported. See the paper for benchmark definitions and evaluation protocols.

## Installation

Requires **Python 3.10+** and **PyTorch 2.4+**. The supplied training configurations use CUDA and BF16.

```bash
conda create -n pde-jepa python=3.12 -y
conda activate pde-jepa

git clone https://github.com/Tanpig-X/PDE-JEPA.git
cd PDE-JEPA
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml` and installed into the active environment. Editable installation uses the code in this repository.

For optional Weights & Biases logging, install `.[logging]` and enable logging in the configuration.

## Data

The shared HDF5 loader supports the following layouts:

| Field | Shape |
|---|---|
| 2D trajectories | `states: [N, C, H, W, T]` |
| 1D trajectories | `states: [N, C, X, T]` |
| Physical parameters | `[N, P]` or `[N, T-1, P]` |

Each file contains a trajectory group. Set field channels and normalization in the configuration, and use `data.condition_key` for parameter dataset names. Fit normalization statistics on the ID training split.

The Vorticity recipe expects:

```text
$DATA_ROOT/vorticity/train.h5
$DATA_ROOT/vorticity/val.h5
$DATA_ROOT/vorticity_ood/val.h5
```

Its `states` have shape `[N, 1, 128, 128, 30]`; `mu` stores viscosity. Fields use mean `0` and standard deviation `3.737`. Datasets and pretrained weights are not bundled.

## Training

The repository provides shared training modules and an end-to-end **Vorticity** configuration. Run all four stages from the repository root:

```bash
export DATA_ROOT=/path/to/data
export OUTPUT_ROOT=/path/to/pde_jepa_runs
export CUDA_VISIBLE_DEVICES=0,1

bash scripts/train_vorticity.sh all
```

Stage 1 includes a separate learning-rate cooldown, giving five configuration files:

| Stage | Configuration |
|---|---|
| 1 · Pretraining | [stage1_pretrain.yaml](configs/vorticity/stage1_pretrain.yaml) |
| 1 · Cooldown | [stage1_cooldown.yaml](configs/vorticity/stage1_cooldown.yaml) |
| 2 · PAG | [stage2_pag.yaml](configs/vorticity/stage2_pag.yaml) |
| 3 · PSP | [stage3_psp.yaml](configs/vorticity/stage3_psp.yaml) |
| 4 · Decoder | [stage4_decoder.yaml](configs/vorticity/stage4_decoder.yaml) |

Replace `all` with `pretrain`, `cooldown`, `pag`, `psp`, or `decoder` to run one stage. Set `NPROC_PER_NODE=1` for one GPU. Checkpoints and logs are written under `OUTPUT_ROOT`.

### Other PDEs

Available predictor task keys are `vorticity`, `advection`, `burgers`, `heat`, `wave_b`, `wave2d`, `grayscott`, and `combined`. Use a task-specific configuration with the shared entry point:

```bash
python -m pde_jepa.train --stage psp --config /path/to/stage3_psp.yaml
```

Set `model.task`, data geometry, physical conditions, and training statistics for the selected PDE. The same entry point accepts all five stage names above. Only Vorticity configurations are supplied; other tasks require their own configuration files.

## Checkpoint evaluation

Evaluate a trained model with its matching encoder, PAG, PSP, and decoder paths in the configuration:

```bash
python -m pde_jepa.train --stage decoder \
  --config configs/vorticity/stage4_decoder.yaml \
  --evaluate-only --checkpoint "$OUTPUT_ROOT/decoder/best.pt" --split ood
```

Use `--split val` for ID validation. Metrics are saved to `evaluation_ood.json` or `evaluation_val.json` in the configured output directory.

## Citation

```bibtex
@misc{tan2026pdejepa,
  title={PDE-JEPA: Predictive Representation Learning of Latent Dynamics Modeling for Parametric PDEs},
  author={Zhentao Tan and Jianrong Zhang and Ruijie Quan and Yi Yang},
  year={2026},
  eprint={2609.34715},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.34715}
}
```

## License

Distributed under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for code attribution.
