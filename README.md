# Health-ML-Bias-Experiments: Causal Framework for Fair Cardiac Prediction

This repository contains the experimental pipeline for my MRes dissertation: **"Causal Framework for Auditing and Mitigating Unfair Bias in Cardiac Prediction Models."** The project addresses a critical challenge in clinical AI: distinguishing between **legitimate biological differences** (necessary for diagnostic accuracy) and **unfair sociological bias** (which leads to health disparities).

---

## Core Research Questions
1.  **RQ1 (Disparity):** What predictive performance disparities exist in state-of-the-art Atrial Fibrillation (AF) models when stratified by ethnic and sex subgroups?
2.  **RQ2 (Causality):** How can causal analysis and counterfactuals identify the unfair pathways driving these inequities?
3.  **RQ3 (Mitigation):** Can a disentangled latent variable approach mitigate bias while preserving clinical utility?

---

## Causal Framework: Disentangled Latent Variables
We make the conceptual assumption that the impact of protected attributes in clinical applications can be
considered as the combination of their biological relevance to health outcomes and their sociological effect
on health processes. We represent this concept by decomposing the protected attribute ($A$) into two distinct
causal components:

* **Biological component ($A_{bio}$):** Physiological, genetic, or ancestral factors clinically relevant to the health outcome
* **Sociological component ($A_{soc}$):** Systemic biases in how patients are perceived or treated within the healthcare system, as well as historical disparities

### Feature Mapping
To operationalize this, observed features are categorised based on their relationship with these components: 
* **$X_{corr}$ (Baseline clinical covariates):** Features preceding the sociological component, such as biological markers or baseline clinical measurements
* **$X_{desc}$ (Sociologically-influenced features):** Descendants of the sociological component susceptible to systemic bias, such as clinician interpretations or reported symptoms

### Methodology: CEVAE-HE
We propose the Causal Effect Variational Autoencoder for Health Equity (CEVAE-HE) architecture, built upon the Disentangled Causal Effect Variational Autoencoder (DCEVAE) (Kim et al., 2021). The neural network disentangles exogenous uncertainty into two latent variables:
1.  **$U_{desc}$:** Exogenous factors independent of the protected attribute
2.  **$U_{corr}$:** Exogenous factors correlated with the protected attribute without causality

![CEVAE-HE network in training](/media/DCEVAE-Training.jpg)
*A. Min phase: the network's objective is to minimise the total VAE loss. B. Max phase: the discriminator's objective is to maximise its ability to discriminate between real and permuted samples.*

The architecture comprises two primary encoder networks that map the observed features to the disentangled latent spaces $U_desc$ and $U_corr$, alongside decoder networks tasked with reconstructing the original features and predicting the outcome. To ensure effective disentanglement, the model incorporates an adversarial discriminator network that penalises any mutual information between the "fair" latent space and the protected attribute.

The CEVAE-HE model operates on a min-max objective structure. The encoders and decoders are trained to minimize a modified Evidence Lower Bound (ELBO) for accurate data reconstruction, alongside a Total Correlation loss that forces $U_{desc}$ and $U_{corr}$ to remain statistically independent to each other and $U_desc$ to be independent of the protected attribute, and a Fairness loss maximising counterfactual fairness. Simultaneously, the discriminator is adversarially trained to maximize its ability to distinguish samples from the joint distribution $q(a,u_{desc},u_{corr})$ from samples drawn from the marginal distributions $q(a,u_{desc})q(u_{corr})$ (simulated with shuffled $(a, [u_{desc},u_{corr}])$ pairs).


---

## Experimental Pipeline
The project is structured in two distinct phases:
1.  **Validation Pilot:** Implementation on the UCI dataset using  a **Random Forest** classifier as baseline. 
2.  **Primary Analysis:** Scaling the pipeline to the **UK Biobank (UKB)** to audit a state-of-the-art AF prediction model (99 features, N≈42,500).

### Advanced Auditing Tools
* **Dual-Counterfactual Test:** Distinguishes between "Pathway-specific bias" (unfair) and "Direct bias" by simulating sociological attribute flips vs. total attribute flips. 
* **Total Effect (TE) Error:** Measures the difference between observed and estimated changes in outcome probability to validate the causal model's integrity.


---

## Key References
* Kim, H. et al. (2021). "DCEVAE: Disentangled Causal Effect Variational Autoencoder." [cite: 116]
* Kusner, M. J. et al. (2017). "Counterfactual Fairness." [cite: 16]
* Straw, I. et al. (2024). "Algorithmic Bias in Cardiovascular Disease." [cite: 54]