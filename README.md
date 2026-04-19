This repository contains the experimental pipeline for my MRes dissertation: **"Causal Framework for Auditing and Mitigating Unfair Bias in Machine Learning Models for Atrial Fibrillation Prediction"**. 

AI models internalise systemic inequities from their training data, leading to performance disparities—such as higher False Negative Rates (FNR)—in some subpopulations (e.g. ethnic minorities, women). This project addresses a challenge specific to clinical AI: distinguishing between legitimate biological differences (necessary for diagnostic accuracy) and unfair sociological bias (which leads to health disparities) between subgroups and individuals.

---
# Project overview
Current state-of-the-art AF prediction models trained on electronic Health Record (EHR) data achieve high global accuracy but mask predictive failures for protected subgroups. Our proposed framework utilises causal modeling and counterfactual inference to disentangle "fair" biological signals from "unfair" sociological pathways (e.g. systemic under-diagnosis or inequitable access to care).

## Research proposal

[Full rationale and protocol](https://drive.google.com/file/d/14BSkQFsATUxE-e06ByTi629HocjdGgJ3/view?usp=sharing) 

## Research question and objectives
To what extent can a disentangled causal framework separate biological from sociological bias in EHR data to achieve counterfactual fairness in machine learning models for Atrial Fibrillation prediction?

1. **Establish a performance and bias baseline** by replicating a state-of-the-art "fairness-unaware" ML model for AF prediction on the UK Biobank (UKB).
2. **Disentangle fair and unfair influences of the sensitive attribute** by developing a CEVAE-HE architecture that projects EHR features into a fair latent space.
3. **Validate the causal feature-to-pathway assignments** by using a pathway sensitivity analysisand a dual-objective decision-making framework to balance fairness with clinical utility.
4. **Mitigate individual and group-level disparities** by leveraging the fair latent space to train fair predictive models.
5. **Evaluate the clinical value and robustness of the framework** by evaluating the utility-fairness trade-off, specifically measuring the reduction in False Negative Rate (FNR) disparity relative to overall predictive utility, measured by the Area Under the Receiver Operating Characteristic Curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC)

---
# Methodology

## Disentangled health SCM

We make the assumption that the impact of a sensitive attribute $S$ (e.g. race or sex) on clinical outcomes $Y$ is a composite of biological relevance ($S_{bio}$) and sociological influence ($S_{soc}$).

We partition Electronic Health Record (EHR) features into three causal pathways:

- $X_{desc}$ (Sociologically-influenced): Features susceptible to systemic bias (e.g. clinician interpretation, self-reported symptoms).
- $X_{corr}$ (Baseline clinical covariates): Features correlated with the biological sensitive attribute (e.g. genetic markers, baseline physiological measurements).
- $X_{ind}$: Variables independent of the sensitive attribute.

To account for variation not explained by observed variables, we postulate two latent factors: $U_{desc}$ (parent of $X_{desc}$) and $U_{corr}$ (parent of $X_{corr}$), and we assume the existence of a global health confounder (H) that influences both $U_{desc}$ and $U_{corr}$, acknowledging that both latent variables represent facets of a single patient’s health state.

![Structural Causal Model for disentangled bias of sensitive attribute](/media/health-scm.png)

*Structural Causal Model representing the causal relationships between the sensitive attribute $S$, observed clinical features $X$ and latent variables $U$ on the clinical outcome $Y$.*

## Neural network architecture

We propose the Causal Effect Variational Autoencoder for Health Equity (CEVAE-HE) architecture, built upon the Disentangled Causal Effect Variational Autoencoder (DCEVAE) (Kim et al., 2021), to approximate the latent space  and facilitate counterfactual generation.

![CEVAE-HE network in training](/media/DCEVAE-Training.jpg)
*A. Min phase: the network's objective is to minimise the total VAE loss. B. Max phase: the discriminator's objective is to maximise its ability to discriminate between real and permuted samples.*

---

# Project structure

~~~
├── configs/                # Feature mapping, SCM definitions, and CEVAE-HE scaling parameters (JSON)
├── notebooks/              # Data analysis and result visualisation
├── src/                    
│   ├── cevaehe/            # CEVAE-HE implementation (Model, Train, Test)
│   ├── classifiers/        # Classifier model training for CVD prediction
│   ├── config.py           # Paths and default params config
│   ├── metrics.py          # Fairness and utility metrics
│   ├── bootstrap.py        # MAIN SCRIPT for baseline and fair classifiers training and evaluation
│   ├── finetuning.py       # MAIN SCRIPT for CEVAE-HE scaling parameters finetuning
│   └── pathway_audit.py    # MAIN SCRIPT for Stochastic Pathway Sensitivity audit
├── requirements.txt        # Project dependencies
└── *.sh                    # Execution scripts
~~~

---

# Getting started

## Prerequisites

- Python 3.10+
- To run the protocol on the pilot study, download the ["Heart Disease" dataset](https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive) (Siddharta, 2020) into a `data/` folder at the root of the project

## Installation

~~~
git clone https://github.com/lusyo/health-ml-bias-experiments.git
cd health-ml-bias-experiments
pip install -r requirements.txt
~~~

## Running the pipeline

### 0. Data pre-processing

Run `notebooks/UCI_data_preparation.ipynb` on the pilot dataset

### 1. Stochastic Pathway Sensitivity analysis

**Goal:** Find the optimal $(X_{desc}, X_{corr}, X_{ind})$ feature assignment maximising causal fidelity (OBJ 1: minimise Total Effect error between the observed disparity in the dataset and the CEVAE-HE estimated disparity after reconstruction) and counterfactual stability (OBJ 2: minimise Mean Absolute Counterfactual Error (MACE), i.e. the number of individuals for whom the prediction flips in the counterfactual world). 

1. Execute `run_sps_audit.sh`
2. Analyse results in `notebooks/sps_audit_result.ipynb`: look for the configurations on the TE Error / MACE Pareto Frontier
3. Update `config/uci_scm_config.json` with the optimal pathway configuration

***To skip this step**, run next step with a pre-set pathway configuration, e.g. `config/uci_scm_config.json`*

### 2. CEVAE-HE scaling parameters fine tuning

Finetuning is done in two steps, to minimise compute costs. We keep `pred_a` fixed to finetune other parameters relatively to it:
  1. **Fairness parameters:** exploring values for `fair_b` and `tc_b` for fixed `pred_a`, `corr_a`, and `desc_a`
      - Execute `run_finetuning.sh` with `--mapping "configs/YOUR_CHOSEN_PATHWAY_CONFIG.json"` and `--param_space "configs/fairness_param_space.json"`
      - Analyse results with `notebooks/finetuning_fairness_results.ipynb`: look for configurations on the Pareto Frontier for TE Error vs. $U_{desc}/X_{desc}$ Sensitive Utility Delta 
  2. **Reconstruction parameters:** exploring values for `corr_a` and `desc_a` with best `fair_b` and `tc_b` and pre-fixed `pred_a`
      - Execute `run_finetuning.sh` with `--mapping "configs/YOUR_CHOSEN_PATHWAY_CONFIG.json"` and `--param_space "configs/recon_param_space.json"`
      - Analyse results with `notebooks/finetuning_recon_results.ipynb`: look for configurations on the Pareto Frontier for TE Error vs. $U_{desc}/X_{desc}$ Sensitive Utility Delta 

***To skip this step**, run next step with the default parameters*

### 3. CEVAE-HE training and latent / counterfactual generation

- **STEP A:** Execute `run_cevaehe.sh` with your chosen pathway configuration, scaling parameters and with your chosen experiment name `--exp_name "STEP_A_EXP_NAME"` *(Recommendation: avoid changing other parameters such as learning rates and warm up periods, which have been globally optimised upstream of this pipeline)*
- **STEP B:** Execute `run_bootstrap.sh` with your chosen pathway configuration and scaling parameters, and with `--cevaehe "STEP_A_EXP_NAME_cevaehe.pth" --cf_dataset "STEP_A_EXP_NAME/counterfactuals.csv" --latent_dataset "STEP_A_EXP_NAME/latent_spaces.csv"`
- **RESULTS:** Analyse results with `notebooks/results.ipynb`

---

# Key references

- Kim, H., S.Shin, J.Jang, K.Song, W.Joo, W.Kang, and I.-C. Moon (2021). “Counterfactual Fairness with Disentangled Causal Effect Variational Autoencoder”. In: Proceedings of the AAAI Conference on Artificial Intelligence 35(9), pp. 8128–8136. doi: 10.1609/aaai.v35i9.16990.
- L. Siau, 2026, *Causal Framework for Auditing and Mitigating Unfair Bias in ML Models for AF Prediction*, UCL MRes Dissertation