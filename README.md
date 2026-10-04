# CAD Transcriptomics MVP

A hackathon-ready research prototype for exploring coronary-artery-disease-associated transcriptomic patterns from human RNA-seq expression matrices.

## What it does

- Accepts CSV/TSV gene-expression matrices.
- Detects common gene-as-row or sample-as-row layouts.
- Optional CAD/control metadata.
- Generates PCA visualization.
- Calculates a transparent prototype CAD molecular similarity score.
- Trains an interpretable logistic-regression classifier when labels are supplied.
- Reports cross-validated AUROC.
- Shows model coefficients.
- Summarizes predefined molecular programs: inflammation, immune recruitment, T-cell program, endothelial activation, oxidative stress, lipid handling and matrix remodeling.
- Includes a validation plan for GSE180081, GSE180082 and GSE221911.

## Important scientific limitation

This repository is a **research/hackathon prototype**. It is not a diagnostic medical device. The included molecular score is not a validated clinical probability, and the app must not be used to diagnose CAD, predict future events, determine treatment, or make patient-management decisions.

## Recommended data strategy

Use:

1. GSE180081 — discovery.
2. GSE180082 — validation.
3. GSE221911 — additional whole-blood RNA-seq / stenosis-oriented analysis.

Do not concatenate datasets blindly. Freeze preprocessing, feature selection and the model on the discovery cohort before external validation.

The GSE221911 GEO record describes whole-blood Illumina RNA-seq and states that raw sequencing data are not provided publicly because the raw data contain identifiable sequence information. Use the processed expression files supplied by GEO for the prototype.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/app.py
```

Open the local URL shown by Streamlit.

## Train a saved model

```bash
python scripts/train_model.py \
  --expression data/expression.csv \
  --metadata data/metadata.csv \
  --out models/cad_model.joblib
```

## Expected input

Expression matrix:

```text
Gene,Sample1,Sample2,Sample3
IL6,12.4,7.1,21.2
CXCL8,4.5,2.3,8.9
...
```

or sample-oriented:

```text
Sample,IL6,CXCL8,ICAM1
S1,12.4,4.5,8.2
S2,7.1,2.3,4.1
...
```

Metadata:

```text
Sample,CAD
S1,CAD
S2,Control
...
```

## Suggested next research steps

- Replace the prototype signature with a properly nested-CV-derived signature.
- Use DESeq2/variance-stabilizing transformation for count data.
- Correct/assess batch effects without test-set leakage.
- Validate against a fully held-out cohort.
- Add calibration and decision-curve analysis.
- Separately model stenosis severity only where the cohort provides appropriate clinical labels.
- Add gene-set scoring using GSVA/ssGSEA and curated Reactome/GO gene sets.
- Add cell-type deconvolution for whole blood.
- Add provenance/versioning for every dataset and preprocessing step.
