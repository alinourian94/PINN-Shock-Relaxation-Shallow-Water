# PINN-Shock-Relaxation-Shallow-Water
Physics-Informed Neural Network (PINN) framework for hydrodynamic and shallow water flow problems, including shock-dominated regimes. The repository contains the research implementation used in the associated study, featuring locally relaxed momentum constraints, adaptive weighting, and sequential training for discontinuous flows.

Overview:
This repository contains the research implementation of Physics-Informed Neural Networks (PINNs) used for solving shallow water and hydrodynamic flow problems, including shock-dominated regimes such as dam-break flows. The study focuses on improving solution accuracy and training behavior near discontinuities rather than computational efficiency.

The proposed framework integrates:
Locally relaxed momentum constraints,
Rational (physically-motivated) loss weighting,
Adaptive sampling near discontinuities,
Sequential (causality-aware) training in time,
Two-stage optimization (ADAM → LBFGS for shock cases).

These implementations reflect the final research configuration used to generate the published results. Due to the stochastic nature of neural network training and sensitivity to hyperparameters, minor variations in results may occur.
The repository is intended to provide methodological transparency, not an optimized or fully automated simulation framework.

Requirements:
Python 3.9+ ,
TensorFlow / SciANN ,
NumPy ,
SciPy ,
Matplotlib.

These codes are research prototypes and may require adjustment for different environments.
Computational efficiency was not the focus of this study.
Hyperparameters are selected based on extensive empirical testing reported in the paper.

