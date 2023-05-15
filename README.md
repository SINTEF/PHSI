# Pseudo-Hamiltonian system identification

The folder named "phsi" contains code for the pseudo-Hamiltonian system identification model, and the notebooks contain comparisons between the models where plots can be reproduced. 

# Abstract

This thesis concerns the application of physics-informed machine learning to dynamical systems that can be represented as first-order ordinary differential equations. Current system identification models struggle to learn energy-preserving dynamical systems where damping and external forces affect the training data.  We will tackle this problem by letting our model assume a pseudo-Hamiltonian structure, meaning we learn the inner and outer dynamics separately. We use system identification to learn the inner dynamics, while a neural network will generally be employed to learn the external forces. But, we also explore the possibility of learning the external forces through system identification. Furthermore, we introduce an integration scheme for training the model that attempts to handle noisy data.
