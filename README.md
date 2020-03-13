# HEPAutoencoders

Compressing ATLAS trigger jet events data using autoencoders. The data is available [here](https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr). The data consists of 4D Lorentz vectors which are to be compressed down to 3D.

### Requirements
* PyTorch >= 0.4.0

### Optional
* TensorFlow 2.1.0 and TensorBoard 2.1.1 for live visualization of the training process

### Approach
The approach is to scale the data between \[0,1\] instead of normalizing with 0 mean and unit variance. Having the data in range \[0,1\] gives more choice of activation functions. In the network only fully connected layers are used. The architecture is `4-512-256-128-3-128-256-412-4`. The activation function of choice is LeakyReLU. The network is trained to reconstruct the 4 original values using MSELoss.

