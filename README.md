# IE7615 Generative Project — Flickr30k Text Embedding Baseline

**Course:** IE7615 Neural Networks & Deep Learning — Generative Project
**Group 9 · Milestone 1**

Baseline for text-conditioned image generation with Stable Diffusion 1.5. This
milestone covers dataset preparation, the frozen CLIP text-to-embedding pipeline,
and five pretrained prompt-to-image sample tests.

## Team

| Member | Role |
| --- | --- |
| Xiaoxiao Song | Coding |
| Sneha Mishra | Modified the code, verified results, and drafted the proposal |
| Ching-Chieh Lin | Coding |
| Che-Wei Lee | Coding |

## Repository structure

```
.
├── Milestone 1.ipynb         # Baseline notebook: data prep, CLIP embeddings, 5 sample generations
├── Milestone 1 Proposal.pdf  # 1-page proposal (dataset, model, roles, sample tests + results)
├── requirements.txt          # Pinned dependencies
├── .gitignore
└── README.md
```

Running the notebook also creates an `artifacts/` directory (dataset card,
JSONL manifests, saved embeddings, sample figures, generated images, and an
execution summary). These outputs are reproducible and are git-ignored.

## Dataset

- **Source:** Flickr30k. The Hugging Face mirror stores the whole corpus
  (31,783 images) under a single `test` split.
- **Sampling:** a seeded (seed 42) uniform random sample is drawn across the full
  31,783-image corpus, then cleaned down to **3,000** image–caption pairs (within
  the required 1k–5k range).
- **Preprocessing:** Unicode (NFKC) caption normalization, removal of empty and
  duplicate captions, EXIF-orientation correction, pixel-hash deduplication, and
  512×512 RGB center crops. One caption per image.
- **Splits:** custom image-disjoint 2400 / 300 / 300 (train / val / test). A custom
  split is used rather than the standard Karpathy split because Karpathy's fixed
  1,014 / 1,000 validation / test sets are disproportionately large for a 3,000-pair
  subset.

## Model

Stable Diffusion 1.5 components: the paired CLIP text encoder
(77 × 768 token embeddings), a latent-space U-Net, and a VAE. All weights are
kept frozen; this milestone runs the pretrained pipeline only.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The notebook (cell 1) also checks for and installs any missing dependencies.
Install the CUDA build of `torch`/`torchvision` instead of the CPU wheels if a
GPU is available.

## How to run

Open and run `Milestone 1.ipynb` top to bottom.

> **Cold-start note:** the first run downloads Stable Diffusion 1.5 (~4–5 GB) and,
> if the raw dataset is absent, streams the Flickr30k subset. On CPU the five
> generations take roughly 35–40 minutes, so a cold run is about **50–60 minutes**
> depending on network speed. On a GPU the compute drops to a few minutes.

## Results (executed run)

- Environment: CPU — PyTorch 2.6.0, Diffusers 0.35.1, Transformers 4.49.0.
- 3,000 captions encoded to finite, reproducible float32 77 × 768 sequences;
  no caption truncated.
- Five prompt-to-image tests (DDIM, 30 steps, CFG 7.5, seeds 42–46), each saved
  as an RGB 512×512 image.
- **18 / 18 validation checks passed** (split isolation, finite/repeatable
  embeddings, empty and long prompts, tensor reload, saved images, and
  prompt-seed mapping).


