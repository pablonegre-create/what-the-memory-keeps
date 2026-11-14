# What the memory keeps

Code for the paper *What the memory keeps: exact temporal explanations of recurrent video
classifiers and their dependence on memory length* (P. Negre, under review).

A video classifier made of a frozen encoder, an LSTM and a linear layer can be explained in time
without approximations. The final cell state is a sum of what each step wrote, discounted by
everything that was forgotten afterwards:

    c_T = sum_t ( prod_{k>t} f_k ) * ( i_t * g_t )

Pushing this through the output gate and the linear readout gives a temporal class activation map
whose scores add up exactly to the logit (minus the bias), together with retention maps
(how much of step t is still in memory at step tau) and causal tests that block memory writes.
The same holds for a GRU (h_T is a sum of (prod z)(1 - z) n) and for each direction of a
bidirectional LSTM.

The repository also contains the experiments of the paper: forget-gate bias initialisation
(memory length), detection, localisation against frame-level annotations, ten explanation
baselines, faithfulness, interventions, paired statistics and the figures.

## Layout

```
src/          models, replay/decomposition, experiments and analysis
configs/      experiment configurations (JSON)
results/      aggregated results reported in the paper (reports, statistics, timing)
figures_src/  scripts that draw the figures from the outputs in out/fig
```

Main modules:

| file | content |
|---|---|
| `src/common.py` | `Model` (LSTM / GRU, uni- or bidirectional), float64 replay of the gates, `decompose` (exact per-step contributions to the final memory and to the logit) |
| `src/fastcam.py` | vectorised temporal CAM: one cuDNN pass for h_t, then all gates at once; equality check against the replay and timing |
| `src/exp.py` | forget-gate bias initialisation (`init_bias`), training and evaluation |
| `src/data.py` | adapters for XD-Violence, UCF-Crime and THUMOS14 (same output structure for all) |
| `src/full.py` | full experiment: detection, memory explanations, gradient / perturbation baselines, faithfulness, write-blocking interventions, randomisation check |
| `src/lstmx.py` | LSTM-specific baselines: Murdoch & Szlam (cell and additive), LRP for LSTM, contextual decomposition |
| `src/full_methods.py` | model-agnostic baselines (occlusion, Grad x Input, Integrated Gradients, sampled Shapley) |
| `src/ana.py` | aggregation over seeds; action duration, coverage and tail analyses |
| `src/stats.py` | paired Wilcoxon tests per video with Holm correction |
| `src/p1.py`, `src/p2.py`, `src/p3.py` | initialisation pilot and ablation (memory length vs. detection and localisation) |
| `src/figdata.py`, `src/figsum.py`, `src/fr.py` | data for the figures (examples, memory maps, summaries, video thumbnails) |
| `src/fx.py`, `src/dz.py` | I3D feature extraction for UCF-Crime from the original videos |
| `src/ck.py` | consistency check between features and annotations; duration and coverage statistics |

## Data

The models work on pre-extracted snippet features. Expected layout (relative to the repository root):

```
data/feat/xd_i3d/i3d-features/      XD-Violence I3D RGB features (official release)
data/feat/xd_clip/                   XD-Violence CLIP features (VadCLIP release)
data/feat/ucf_clip/UCFClipFeatures/  UCF-Crime CLIP features (VadCLIP release)
data/feat/ucf_i3d/                   UCF-Crime I3D features, extracted with src/fx.py
data/feat/thumos/thumos/             THUMOS14 I3D features and annotations (ActionFormer release)
data/annot/                          temporal annotations and test lists (VadCLIP release)
```

Only the centre crop of each feature set is used. Sources:

- XD-Violence: <https://roc-ng.github.io/XD-Violence/>
- VadCLIP features and annotation files: <https://github.com/nwpu-zxr/VadCLIP>
- UCF-Crime videos: <https://www.crcv.ucf.edu/projects/real-world/>
- THUMOS14 features: <https://github.com/happyharrycn/actionformer_release>
- I3D for `src/fx.py`: `pytorch_i3d.py` and `rgb_imagenet.pt` from <https://github.com/piergiaj/pytorch-i3d>, placed in `ext/`

## Running

Install the requirements (Python 3.13, PyTorch 2.x with CUDA), then run from the repository root:

```bash
python src/ck.py                                   # check features against annotations

python src/p2.py && python src/p3.py               # initialisation ablation (XD-Violence I3D)

CFG=configs/cfg.json     python src/full.py        # LSTM, 5 datasets x encoders, 5 seeds
CFG=configs/cfg.json     python src/lstmx.py       # LSTM-specific baselines
CFG=configs/cfg.json     python src/ana.py         # aggregation and duration / coverage analyses
CFG=configs/cfg_gru.json python src/full.py        # GRU
CFG=configs/cfg_bi.json  python src/full.py        # bidirectional LSTM
python src/stats.py                                # paired tests
python src/fastcam.py                              # vectorised CAM: equality and timing

python src/figdata.py && python src/figsum.py && python src/fr.py
python figures_src/figs.py  out/fig figures
python figures_src/figs2.py out/stats.md out/full/timing_fastcam_xd_i3d.json figures
python figures_src/ga.py    out/fig figures
```

`configs/cfg_xs.json`, `cfg_gs.json` and `cfg_bs.json` are short smoke runs. TF32 is disabled so
that the decomposition matches the network output to float precision. Figure 1 is drawn in TikZ
(`figures_src/fig1_method.tex`).

## Citation

The reference will be added when the paper is published.
