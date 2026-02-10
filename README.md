PINN-Shock-Relaxation-Shallow-Water

Physics-Informed Neural Network (PINN) framework for shallow-water and hydrodynamic flow problems, including shock-dominated regimes. This repository contains the research implementation used in the associated study, featuring locally relaxed momentum constraints, rational loss weighting, adaptive sampling, and sequential training for discontinuous flows. 

Overview: 
This work focuses on improving solution accuracy and training behavior near discontinuities rather than computational efficiency. The proposed framework integrates: 
Locally relaxed momentum constraints,
Physically-motivated rational loss weighting,
Adaptive sampling near shocks,
Sequential (causality-aware) temporal training,
Two-stage optimization (ADAM → LBFGS for shock cases). 

For shock-containing problems, training is performed in two stages: ADAM for global exploration followed by LBFGS for solution refinement and improved shock resolution. Local relaxation is applied selectively to the momentum equation to enhance stability and accuracy near discontinuities. 

Notes on Experiments: 
The provided scripts correspond to representative research configurations and are not individual final runs. For each method, multiple simulations were performed using different parameters and settings. The reported results and figures in the study were obtained by aggregating these runs and comparing the overall performance of the proposed approaches. 

Due to the stochastic nature of neural network training and sensitivity to hyperparameters, minor variations in results may occur. The repository is intended to provide methodological transparency rather than a fully optimized or automated simulation framework. 
Requirements: 
Python 3.9+ , 
TensorFlow / SciANN, 
NumPy, 
SciPy, 
Matplotlib. 

Additional Notes: 
These codes are research prototypes and may require adaptation for different environments. 
Computational efficiency was not the focus of this study. 
Hyperparameters were selected based on extensive empirical testing reported in the manuscript. 
