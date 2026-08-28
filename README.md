# Belief Polarization is Not Always Irrational (Replication)

A replication of Jern, Chang, & Kemp (2014), *"Belief Polarization is Not Always Irrational,"* published in *Psychological Review* (121(2), 206–224), with some additional exploratory modeling. By Victor Alexander Moreno, in collaboration with Dr. Michael C. Frank and Ke Fang.

## Overview

Following Jern et al. (2014), participants completed a controlled medical diagnosis task in which they were told that a doctor was trying to diagnose a patient with one of four fictitious diseases. Two — L1 and L2 — were Allozedic diseases, while the other two — Y1 and Y2 — were Hypozedic diseases. They were also told that the patient's treatment depends on which broader category their diagnosis falls into. To establish initial beliefs, participants were shown a chart visualizing the prevalence of each disease among 1,000 previous patients with similar symptoms. Based on this information, they were asked to report their beliefs about which broader disease category the patient's true diagnosis is more likely to fall into.

Participants revised their beliefs after observing the patient's test results, which varied across the Polarization, Moderation, and Control conditions. As in Jern et al. (2014), we examine whether participants' belief updates track a Bayesian, rational-updating account. To do so, this replication reruns the paradigm on a new sample recruited via Prolific, and compares the observed results to the predictions from several iterations of a Bayesian model adapted from the original paper.

## Repository structure

```
├── analysis/                Quarto notebooks for the analysis pipeline
│   ├── JernRep-PowerAnalysis.qmd
│   ├── JernRep-ExperimentalStimuli.qmd
│   ├── JernRep-DataProcessing.qmd
│   ├── JernRep-Statistics.qmd
│   ├── JernRep-ModelPredictions.qmd
│   ├── JernRep-Figures.qmd
│   └── rendered/            Rendered HTML output (gitignored, see below!)
│
├── data/
│   ├── raw/                 Unprocessed Qualtrics exports (full study + pilot)
│   └── processed/           Cleaned dataset + all derived statistical/model outputs
│
├── materials/
│   ├── original_paper/      Jern et al.'s original stimulus charts, story template provided by Dr. Alan Jern
│   ├── stimuli/             Recreated stimulus charts, and a screen-by-screen PDF of what participants in our replication saw
│   └── figures/             Figures generated from our analyses
│
├── writeup/                 Model formalization notes
├── anonymize.py             Strips PII columns from raw Qualtrics exports
├── references.bib           Bibliography used across analysis notebooks
└── renv.lock                R package versions
```

## Analysis

We use [renv](https://rstudio.github.io/renv/) to track R package versions. In the project root:

```r
renv::restore()
```

Please find a breakdown of the analysis notebooks below, with each depending on outputs from those listed before it:

1. **`JernRep-PowerAnalysis.qmd`** : sample size / power planning, ran before data collection.
2. **`JernRep-ExperimentalStimuli.qmd`** : recreates the chart stimuli used in the original study (writes to `materials/stimuli/`).
3. **`JernRep-DataProcessing.qmd`** : cleans the raw Qualtrics export (writes to `data/processed/`).
4. **`JernRep-Statistics.qmd`** : runs primary inferential analyses (writes to `data/processed/`).
5. **`JernRep-ModelPredictions.qmd`** : computes each iteration of our Bayesian model's predictions for comparison against the observed results.
6. **`JernRep-Figures.qmd`** : builds the result figures (writes to `materials/figures/`).

Please render each/any notebook with `quarto render analysis/<file>.qmd`.

## Data

- **`data/raw/`** : the original Qualtrics exports, including data from our full study and the pilot run.
- **`data/processed/`** : the fully processed dataset (`JernRep-Data-Processed-Full.csv`) as well as every downstream statistical and model-prediction output, named to match the associated analysis notebook.

## Materials

- **`materials/original_paper/`** : the original Jern et al. (2014) paper, the stimuli from the original paper, and the diagnostic-reasoning scenario text provided by Dr. Alan Jern.
- **`materials/stimuli/`** : our recreations of the original paper's charts, as well as a PDF visualizing the experiment screen-by-screen as participants saw it.
- **`materials/figures/`** : figures generated from our analyses (produced by `JernRep-Figures.qmd`).

## License

MIT. See [LICENSE](LICENSE).
