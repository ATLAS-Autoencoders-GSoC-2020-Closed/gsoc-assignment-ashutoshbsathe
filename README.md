# HEPAutoencoders

Compressing ATLAS trigger jet events data using autoencoders. The data is available [here](https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr). The data consists of 4D Lorentz vectors which are to be compressed down to 3D.

### Requirements
* PyTorch >= 0.4.0

### Optional
* TensorFlow 2.1.0 and TensorBoard 2.1.1 for live visualization of the training process

### Approach
The approach is to scale the data between \[0,1\] instead of normalizing with 0 mean and unit variance. Having the data in range \[0,1\] gives more choice of activation functions. In the network only fully connected layers are used. The architecture is `4-512-256-128-3-128-256-512-4`. The activation function of choice is LeakyReLU. The network is trained to reconstruct the 4 original values using MSELoss.

### How to Use ?
1. Download the [data](https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr) and put the `.pkl` files in the same directory as this repository
2. Run `train.py`. This will start the training of the AutoEncoder. Note that any changes in hyperparameters need to be changed via modifications in `train.py` itself (at least for now)
3. While the training code is running, you can visualize the losses on TensorBoard
4. After the training is complete, there will be a TensorFlow summary file and checkpoints in the `./runs/XXXXX/` directory(`XXXXX` is the timestamp for start of the training)
5. Now you need to download data from TensorBoard in the form of CSV for tag `test_loss`. Then give path to your CSV file and relevant checkpoint (`.pt`) files in `generate_plots.py` to generate required plots.

