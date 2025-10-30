# A Voting System to Optimize Daily Forest Fire Prediction

## Abstract

Forest-fire prediction using Artificial Intelligence (AI) continues to face major challenges, including (i) the ability to generalize across regions with very different risk profiles, (ii) managing the inherent daily variability and randomness of fire occurrences (including extreme fire days). These factors together have hindered the deployment of dependable prediction systems in operational settings. In this work, we introduce a novel multi-risk modeling framework specifically designed to tackle all two challenges simultaneously. The proposed approach is applied to daily forest-fire prediction across mainland France. We develop a voting-based system that combines the outputs of multiple models trained on signals smoothed with a range of convolutional kernels, capturing both local and seasonal variations. The proposed solution achieves superior performance compared to conventional models, demonstrating improved cross-regional transferability and robustness to daily fluctuations. Notably, it significantly enhances prediction skill for the rare but damaging extreme-fire days, where traditional models often fail. Our experiments reveal that using an ensemble of multiple risk models can better capture the complex dynamics of fire risk and provide more reliable guidance for decision-makers.

## Repository structure

| Path | Description |
| --- | --- |
| `supplementary_materials.pdf` | Additional analyses, figures, and experimental results supporting the manuscript. |
| `code/discretization.py` | Implements the risk discretization strategies (K-Means) and preprocessing pipelines (seasonal convolutions, Gaussian/cubic/quartic smoothing, persistence) that prepare the signals and aggregate observations by territory. |
| `code/pytorch_models.py` | Defines the PyTorch architectures (MLP, dilated CNN, GRU, GNN) and their temporal decoders used to model the risk dynamics from spatio-temporal sequences. |
| `code/sklearn_api_model.py` | Provides a scikit-learn compatible API to train, tune, and evaluate classic ensembles, metrics. Provide the code for the voting system, a similar class can be employed for pytorch models |

## Voting system
The system trains each model on every risk-level configuration, modified by filtering and subsequently clustered using K-Means, and then performs voting over the predicted class probabilities. The primary limitation of this approach is the computational cost: training may require several days or even weeks depending on the complexity of the model. To experiment with your own data, we recommend starting with a simpler model such as logistic regression, which already provides highly satisfactory results.

## Typical usage

1. **Preprocessing and discretization** – Functions in `code/discretization.py` transform the input series (incident counts, meteorological signals) into consistent risk classes per region.
2. **Model training** – PyTorch architectures or scikit-learn ensembles are trained in parallel on the same feature sets to leverage the complementarity between neural and statistical approaches.
3. **Ensemble aggregation** – Normalized outputs are fused via the shared voting system to boost daily prediction robustness, especially for extreme events.

For additional mathematical and experimental details, consult `supplementary_materials.pdf`.
