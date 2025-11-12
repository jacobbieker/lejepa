# LeJEPA
**Lean Joint-Embedding Predictive Architecture (LeJEPA): Provable and Scalable Self-Supervised Learning Without the Heuristics**
[GitHub Repository](https://github.com/rbalestr-lab/lejepa)  
[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
---


## Demo

<img src="eval/output1.gif" controls width="400">
<img src="eval/output2.gif" controls width="400">
<img src="eval/output3.gif" controls width="400">
<table>
  <tr>
    <td><img src="eval/n01818515_919_original.png" width="200"/></td>
    <td><img src="eval/n01818515_919_pca.png" width="200"/></td>
  </tr>
  <tr>
    <td><img src="eval/n01818515_14304_original.png" width="200"/></td>
    <td><img src="eval/n01818515_14304_pca.png" width="200"/></td>
  </tr>
</table>

| shots | model                  | params | pretrain | epochs | DTD      | aircr.   | cars     | cifar10  | cifar100 | flowers102 | food     | pets     | avg.    |
|-------|------------------------|--------|----------|--------|----------|----------|----------|----------|----------|------------|----------|----------|---------|
| 1     | LeJEPA ViT-L           | 304M   | IN-1K    | 100    | **33.21**| 9.37     | 3.40     | 51.65    | 27.01    | 48.53      | 17.14    | 46.11    | 29.55   |
| 1     | LeJEPA ConvNeXtV2-H    | 660M   | IN-1K    | 100    | 32.15    | 8.07     | 4.28     | 50.95    | **31.48**| **48.74**  | **17.95**| **58.98**| **31.58**|
| 1     | I-JEPA ViT-H           | 632M   | IN-1K    | 300    | 27.71    | **9.86** | **4.33** | **56.52**| 30.58    | 44.69      | 14.53    | 53.38    | 30.20   |
| 10    | LeJEPA ViT-L           | 304M   | IN-1K    | 100    | **64.72**| **35.25**| 22.25    | 85.15    | 59.77    | **92.53**  | **50.90**| 77.00    | **60.95**|
| 10    | LeJEPA ConvNeXtV2-H    | 660M   | IN-1K    | 100    | 61.84    | 30.67    | **24.46**| 85.74    | 63.29    | 91.78      | 49.32    | 78.53    | 60.70   |
| 10    | I-JEPA ViT-H           | 632M   | IN-1K    | 300    | 57.68    | 33.82    | 21.96    | **88.77**| **66.42**| 88.24      | 43.97    | **83.23**| 60.51   |
| all   | LeJEPA ViT-L           | 304M   | IN-1K    | 100    | **78.30**| 57.01    | **57.28**| 96.50    | 83.71    | **91.21**  | **82.05**| 89.74    | **79.48**|
| all   | LeJEPA ConvNeXtV2-H    | 660M   | IN-1K    | 100    | 76.60    | 52.99    | 54.88    | 96.15    | 81.34    | 91.11      | 77.64    | 89.76    | 77.56   |
| all   | I-JEPA ViT-H           | 632M   | IN-1K    | 300    | 73.32    | **56.61**| 54.47    | **97.54**| **86.42**| 86.47      | 81.02    | **92.11**| 78.50   |

## Overview
LeJEPA is a lean, scalable, and theoretically grounded framework for self-supervised representation learning, based on Joint-Embedding Predictive Architectures (JEPAs). LeJEPA introduces **Sketched Isotropic Gaussian Regularization (SIGReg)**, a novel objective that constrains learned embeddings to an optimal isotropic Gaussian distribution, minimizing downstream prediction risk.
**Key Features:**
- Single trade-off hyperparameter
- Linear time and memory complexity
- Stable training across architectures and domains
- Heuristics-free implementation (no stop-gradient, teacher–student, or schedulers)
- Distributed training-friendly codebase (~50 lines of core code)
- State-of-the-art results across 10+ datasets and 60+ architectures
---
## Installation
LeJEPA is built on [PyTorch](https://pytorch.org/) and standard scientific Python libraries (e.g., NumPy). For rapid experimentation, we provide a pretraining skeleton script using `stable_pretraining`, a PyTorch Lightning wrapper. The core SIGReg loss can be integrated into any pretraining codebase.
**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy
- (Optional) `stable_pretraining` for provided training scripts
**Install via pip:**
```bash
pip install lejepa
```

## Quick Start: Using SIGReg

LeJEPA provides a variety of univariate and multivariate statistical tests for regularizing embeddings. Here is a minimal example using the SIGReg loss:
```
import lejepa

# Choose a univariate test (Epps-Pulley in this example)
univariate_test = lejepa.univariate.EppsPulley(num_points=17)

# Create the multivariate slicing test
loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test, 
    num_slices=1024
)

# Compute the loss (embeddings: [num_samples, num_dims])
loss = loss_fn(embeddings)
loss.backward()
```


## Citation
If you use LeJEPA in your research, please cite:

```
@misc{balestriero2025lejepaprovablescalableselfsupervised,
      title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}, 
      author={Randall Balestriero and Yann LeCun},
      year={2025},
      eprint={2511.08544},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.08544}, 
}
```

## Contact & Contributions
We welcome issues, feature requests, and pull requests!
For questions or collaborations, please contact rbalestr@brown.edu

