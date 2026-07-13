# AION photo-z submission — how to run

This repo is the LSST-DESC PZ data challenge (`LSSTDESC/pz_data_challenge`) with
an AION-based submission named **`aion`** added on top.

## Files added for this submission
- `aion_pz.py` — the estimators: (a) primary = frozen AION-1 embeddings + a
  small MLP head → p(z); (b) zero-shot baseline = AION's native generative p(z)
  via `target_modality=Z`.
- `tests/test_aion.py` — the six required challenge entry points, wired to `aion_pz`.
- `scripts/run_aion_zero_shot.py` — CLI to run the zero-shot baseline.
- `requirements_aion.txt` — extra deps (`polymathic-aion`, `torch`, `scikit-learn`, `joblib`).
- `.github/workflows/submit_aion.yaml` — CI workflow.

## Method (short version)
AION was trained on Legacy Survey / HSC / DESI / Gaia, **not LSST**, and has no
LSST tokenizer. We map LSST `g,r,i,z,y` magnitudes onto AION's HSC magnitude
modalities to obtain embeddings, then concatenate the raw LSST `u` + Roman bands
(which AION cannot ingest) and train an MLP classifier over a redshift grid
(`z ∈ [0, 3]`, 301 bins). Softmax over bins = the per-object p(z), written as a
`qp` interpolated ensemble with `object_id` + `zmode` ancil.

## One-time setup (on the GPU machine)
```bash
cd pz_data_challenge
python -m venv .venv && source .venv/bin/activate   # Python >= 3.13
pip install .           # base challenge package (qp, tables_io, rail, sklearn)
pip install .[dev]      # pytest etc.
pip install -r requirements_aion.txt   # AION + torch (use the CUDA wheel for your box)
```

## Download the public data (~large, from NERSC)
```bash
python scripts/download_public.py     # extracts to ./public and tests/public
```
Files look like `pz_challenge_taskset_1_cardinal_training_1yr.hdf5`
(sim ∈ {cardinal, flagship}, scenario ∈ {1yr, 10yr}).

## Smoke-test the pipeline on one file before wiring the full harness
```python
import aion_pz
aion_pz.train_and_estimate(
    "public/pz_challenge_taskset_1_cardinal_training_1yr.hdf5",
    "public/pz_challenge_taskset_1_cardinal_test_1yr.hdf5",
    "out_taskset1_cardinal_1yr.hdf5",
    save_model_to="model_taskset1_cardinal_1yr.pkl",
)
```

## Zero-shot baseline (no training)
AION emits a p(z) directly from photometry. The redshift grid is read from the
model's own Z codec (`codec.quantizer.codebook`), so no manual bin↔z mapping is
guessed.
```bash
python scripts/run_aion_zero_shot.py \
    public/pz_challenge_taskset_1_cardinal_test_1yr.hdf5 \
    zeroshot_taskset1_cardinal_1yr.hdf5
# --native keeps AION's own bins instead of resampling to the metrics grid
```
Or from Python:
```python
import aion_pz
aion_pz.zero_shot_estimate(
    "public/pz_challenge_taskset_1_cardinal_test_1yr.hdf5",
    "zeroshot_taskset1_cardinal_1yr.hdf5",
)
```
Expect this to under-perform the trained head because AION never saw LSST
filters (and has no u band); it's a comparison point.

## PDF calibration
The binned head produces slightly over-confident PDFs. `train_and_estimate`
holds out 10% of the training set, fits **PIT recalibration** (a monotonic CDF
remap that makes PIT uniform), stores it in the saved model, and applies it to
the test PDFs (the estimation-only path reuses it). Measured on task set 1
(cardinal 1yr, held-out eval): PIT_KS 0.093 → 0.054 (~41% better) with point
metrics unchanged. Temperature scaling was tried and barely helped.
```bash
# reproduce the calibration comparison
python scripts/calibrate_aion_taskset1.py
```

## Run the challenge validation
```bash
# CUDA is auto-detected; force CPU with AION_PZ_DEVICE=cpu
NO_TEARDOWN=1 python -m pytest tests/test_aion.py -k taskset_1 -s
```
Validation passes when each configuration reaches check 7 (object IDs match).

## Score locally against truth
Use `scripts/run_metrics.py` / the `nb/Metrics.ipynb` notebook to compute the
point-estimate (bias, sigmaMAD, outlier rate) and distribution (PIT, CDELoss)
metrics on the training split.

## Still to verify / tune (honest notes)
- `[Unverified]` exact qp API calls (`qp.Ensemble(qp.interp, ...)`, `set_ancil`,
  `write_to`) against the installed `qp-prob` version — confirm on first run.
- `[Inference]` `tables_io.read` may return a grouped dict; `load_catalog`
  handles the common cases but confirm the actual key layout of the HDF5 files.
- The MLP head hyperparameters and the redshift grid (301 bins, zmax=3) are
  starting points — tune against the local metrics.
- For task set 2 (non-representative training), consider reweighting or a
  domain-adaptation step; the current pipeline treats all task sets identically.
- Optionally add a zero-shot AION generative-p(z) baseline (`target_modality=Z`)
  for comparison.
