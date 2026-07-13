# PZ Data Challenge with AION — Project Overview

## The challenge — LSST-DESC Photometric Redshift (PZ) Data Challenge
Estimate each galaxy's **redshift** (its distance / cosmic-age proxy) from photometry alone —
brightness in a few broad filters — instead of slow spectroscopy. You must output a full
probability distribution **p(z)** per object, not just a single number. Data are realistic
*simulated* LSST catalogs: magnitudes in bands `u,g,r,i,z,y` (+ Roman `Y,J,H`), ~100k training
galaxies with true redshift and ~20k test galaxies without. Four task sets of rising difficulty;
we're on **Task Set 1** (representative sample), across two simulations (cardinal, flagship) ×
two survey depths (1yr, 10yr). Submissions are `qp` distribution files scored on accuracy (bias,
scatter, outliers) and calibration (PIT / CDELoss). Part 1 closes 2026-07-17.

## What we did
1. **Studied** the challenge repo and AION-1 (Polymathic's astronomy transformer) to see how AION
   can produce redshifts.
2. **Cloned/merged** the real challenge repo into the working dir and built a submission scaffold
   named `aion` (`aion_pz.py`, `tests/test_aion.py`, requirements, CI workflow).
3. Implemented **two methods**: (a) AION's native zero-shot p(z); (b) frozen AION embeddings + a
   small MLP head trained on the challenge data — the primary approach.
4. **Ran & compared** both on Task Set 1: zero-shot was unusable (bias 0.29, 77% outliers); the
   trained head was strong (bias ~0, scatter 0.027).
5. **Added PDF calibration** (PIT recalibration) — cut miscalibration PIT_KS 0.093 → 0.054 with no
   accuracy loss — and wired it into the pipeline.
6. **Generated all 4 Task-Set-1 outputs**, and validated them with the challenge's own checker:
   all pass checks 1–7.

## Issues & how we overcame them
- **Band mismatch (core problem):** AION was trained on HSC / Legacy Survey, has no LSST tokenizer
  and no u-band. → Mapped LSST `grizy` onto AION's HSC-magnitude inputs for embeddings, and fed the
  raw u + Roman bands straight to the MLP head, which learns to correct the filter offset against
  true redshifts. (This is why zero-shot fails but the head works.)
- **Environment:** the existing `.venv` was Python 3.14 with no pip/torch wheels. → Built an
  isolated 3.13 venv with only the lean deps needed (skipped the heavy full RAIL stack).
- **`safetensors` missing:** AION couldn't load its weights. → Installed it and pinned it in
  `requirements_aion.txt`.
- **Overconfident PDFs:** more training data sharpened point estimates but slightly worsened
  calibration; plain temperature scaling didn't help. → PIT recalibration (monotonic CDF remap)
  fixed it.
- **No CUDA (Apple Silicon):** ran everything on CPU; cheap because scalar-only inputs make AION's
  forward passes tiny.
