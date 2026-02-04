# Health-ML-Bias-Experiments

## Causal Analysis of Bias in Cardiovascular Prediction Models

This repository documents the research and experimental notebooks for my Master of Research (MRes) project on **"Causal Analysis of Ethnic Bias in EHR-based Machine Learning Models for Atrial Fibrillation Prediction"** conducted at UCL. 

[Preliminary brief for the project](https://drive.google.com/file/d/1h9J8RUqvgTgr_NEmLDg2zh0SJjZ40vUO/view?usp=sharing)

This repository focuses on testing the methodology for the initial stages of the project. Experiments will involve the validation and extension of existing bias assessment and mitigation techniques on open datasets, to prepare a  pipeline that will be tested on the UK Biobank dataset.

---

## Core Research Questions

1.  **RQ1 (Disparity):** What predictive performance disparities do state-of-the-art ML models for AF incidence trained on structured EHR data exhibit when predictive accuracy is stratified by ethnic subgroups?

2.  **RQ2 (Causality):** How can causal analysis methods be used to identify and explain the unfair pathways that lead to inequitable prediction accuracies?

3. **RQ3 (Mitigation):** What targeted mitigation strategies can be proposed to address the specific pathways of bias and improve the model’s equity?

---

## Counterfactual fairness

Attempts to mitigate bias in a model by simply removing sensitive attributes from its training, i.e. fairness by unawareness, often fails due to bias 'leaking' through causal relationships between the sensitive attribute and other features retained in the data. The counterfactual fairness approach introduced by Kusner et al. (2017, in Advances in Neural Information Processing Systems, https://proceedings.neurips.cc/paper_files/paper/2017/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) addresses this limitation by deconvoluting the biased observed variables into a fair set of unbiased latent variables. It allows the model to only learn from information that is independent from the protected attribute, and neutralise both direct and proxy bias pathways.

### Notations and definitions

we adopt the following notations consistent with the Pearlian causal framework used by Kusner et al.:

- $S$ **Protected attribute**: The sensitive variable we wish to be fair toward
- $X$ **Observed features**: The set of features available in the dataset (e.g. Blood Pressure, Cholesterol)
- $U$ **Latent (unobserved) variables**: Unobserved variables that are independent of the protected attribute $A$
- $Y$ **Target**: The outcome we are predicting (e.g. Presence of CAD)
- $Y_{S \leftarrow s}$: The value of $Y$ under a counterfactual intervention where $S$ is set to $s$
- $M = (U, V, F)$ a causal model corresponding to the observed data, where $V \equiv X \cup Y \cup S$, and $F$ is the set of structural equations of the model

**Counterfactual Fairness:** A predictor $\hat{Y}$ is counterfactually fair if, for a specific individual, the probability distribution of the prediction is the same in the actual world as it would be in a counterfactual world where their protected attribute (e.g., Sex) was different.

Formally, for any value $a'$ of the protected attribute $A$:$$P(\hat{Y}_{S \leftarrow s} = y \mid X=x, S=s) = P(\hat{Y}_{S \leftarrow s'} = y \mid X=x, S=s')$$

### The experiment

Using the fairness-unaware models trained to predict Cardiovascular Disease in Straw et al. (2024, doi: [10.2196/46936](https://doi.org/10.2196/46936)) as baseline models, we will apply the fairness algorithm proposed by Kusner et al. to train a fair CAD predictor.

**The target bias**
While Kusner focus on mitigating bias on tasks where the protected attribute should have no influence on the target outcome (e.g. sex and exam results), the clinical domain brings a new challenge. Indeed, protected attributes such as sex or race often encompass two variables: the clinically relevant biological attribute which can cause a disease to present differently across individuals, and the sociological attribute which has societal factors that can influence healthcare access, physician perception, diagnosis and care. A clinical outcome might be influenced by the former but should remain independent of the latter.

If we aim for fairness based on the high-level sex attribute, we risk removing legitimate clinical signals and degrading diagnostic accuracy. Therefore, our objective in experiment is to make the model counterfactually fair with regards to **sociological sex**.

**Hypothesis**

By using counterfactual inference to model latent variables that are independent of the protected attribute, we can build a predictor that satisfies counterfactual fairness (i.e. ensuring that an individual’s predicted risk of CAD remains invariant to their sociological sex), while maintaining clinically acceptable predictive performance and reducing the False Negative Rate (FNR) disparity observed in baseline models.