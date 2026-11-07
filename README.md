# IE7615 Generative Project — Flickr30k Text Embedding Baseline

**Course:** IE7615 Neural Networks & Deep Learning — Generative Project
**Group 9 · Milestones 1–2**

Baseline for text-conditioned image generation with Stable Diffusion 1.5. This
milestone covers dataset preparation, the frozen CLIP text-to-embedding pipeline,
and five pretrained prompt-to-image sample tests. Milestone 2 connects the text
encoder to the diffusion U-Net with a hand-written sampling loop, runs the baseline
on test captions, and tunes classifier-free guidance and the noise schedule.

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
├── Milestone 2.ipynb         # Integration check, baseline generation, guidance and noise-schedule tuning
├── requirements.txt          # Pinned dependencies
├── .gitignore
└── README.md
```

Running the notebooks also creates an `artifacts/` directory (dataset card,
JSONL manifests, saved embeddings, sample figures, generated images, and an
execution summary; Milestone 2 writes to `artifacts/milestone2/`). These outputs
are reproducible and are git-ignored.

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

Open and run `Milestone 1.ipynb` top to bottom, then `Milestone 2.ipynb`, which reads
the Milestone 1 outputs from `artifacts/`.

> **Cold-start note:** the first run downloads Stable Diffusion 1.5 (~4–5 GB) and,
> if the raw dataset is absent, streams the Flickr30k subset. On CPU the five
> generations take roughly 35–40 minutes, so a cold run is about **50–60 minutes**
> depending on network speed. On a GPU the compute drops to a few minutes.

## Milestone 1 results (executed run)

- Environment: CPU — PyTorch 2.6.0, Diffusers 0.35.1, Transformers 4.49.0.
- 3,000 captions encoded to finite, reproducible float32 77 × 768 sequences;
  no caption truncated.
- Five prompt-to-image tests (DDIM, 30 steps, CFG 7.5, seeds 42–46), each saved
  as an RGB 512×512 image.
- **18 / 18 validation checks passed** (split isolation, finite/repeatable
  embeddings, empty and long prompts, tensor reload, saved images, and
  prompt-seed mapping).

## Milestone 2 results (executed run)

`Milestone 2.ipynb` was run on CPU (Windows, Python 3.13.7, float32): 68 images in about
100 minutes. All weights stay frozen.

- **Integration:** a hand-written sampling loop (CLIP caption vectors → U-Net cross-attention
  with classifier-free guidance → scheduler → VAE) ends on exactly the same latent as the
  diffusers pipeline. The caption embeddings saved by Milestone 1 match a fresh encoding exactly.
- **Baseline:** 8 test captions with the Milestone 1 settings (DDIM, 30 steps, guidance 7.5):
  mean CLIP score 35.5, against 31.3 for the real Flickr30k photos of the same captions;
  about 105 s per image.
- **Tuning** uses 4 validation captions, so the test captions are not used to choose settings.
  The selection rule was fixed before the run: settings within 0.5 CLIP points of the best count
  as equally good; among those, take the smallest guidance, then the fewest steps.
  - Guidance 0 / 1 / 3 / 5 / 7.5 / 10 / 15 → mean CLIP score 9.3 / 21.4 / 30.5 / 32.7 / 33.8 /
    34.2 / 35.0, so **g = 15** is chosen. The gain above g = 10 comes from one caption and the
    images oversaturate, so the scale should be re-checked with a realism metric (KID or FID).
  - Noise schedule at g = 15: DDIM 30 steps 34.96, **DDIM 15 steps 34.76 (chosen)**, Euler
    ancestral 30 steps 34.76, DPM-Solver++ 15 steps 33.12 (33.64 with Karras sigmas), DDIM 30
    with trailing spacing 33.24, and DDIM 30 with a mismatched linear beta schedule 32.89.
- **First set of generated images:** the 8 test captions with DDIM 15 steps, g = 15. On the 7
  captions with both images it scores 36.3, against 35.3 for the baseline, at about 56 s instead
  of 105 s per image. The safety checker blacked out one image (T5); it is marked in the log and
  left out of the averages.
- **Log:** `artifacts/milestone2/generation_log.csv` has one row per image: settings, seed,
  seconds, CLIP score, safety flag and SHA-256. `artifacts/milestone2/run.json` holds the run
  summary. CLIP score is 100 × the cosine similarity between image and caption under a separate
  CLIP model (LAION ViT-B/32), not the text encoder that guides generation.
- **8 / 8 validation checks passed.**

> **Run-time note:** on CPU, Milestone 2 takes about 100 minutes (about 105 s per 30-step image
> with guidance); on a GPU it takes a few minutes. The first run also downloads the CLIP scoring
> model (~0.6 GB).
