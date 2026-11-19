
## Regularisation concepts 

**Dropout** - During training, randomly zero out a fraction (e.g. 25–50%) of neurons each forward pass, forcing the network to not rely on any single neuron; at inference all units are active (outputs scaled accordingly). It acts like training an implicit ensemble of sub-networks and reduces overfitting.

**BatchNorm** - Normalises each layer's activations to zero mean/unit variance using the current mini-batch's statistics (then rescales with learnable γ, β). This stabilises and speeds up training by keeping activations in a consistent range, allows higher learning rates, and adds slight regularisation via batch noise.

**Early stopping** - Monitor validation loss during training and keep the best model; when validation loss stops improving for a patience window (e.g. 10 epochs), stop training. It prevents the model from overfitting the training set in late epochs and is essentially free regularisation.
