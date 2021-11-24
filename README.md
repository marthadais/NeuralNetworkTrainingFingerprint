# Neural Network Training Fingerprint
Neural Network Training Fingerprint (NNTF) is a visualization approach to analyze the training process of any neural network performing classification [1].

## Requirements

Keras==2.4.3\
matplotlib==3.3.4\
numpy==1.18.5\
pandas==1.2.2\
scipy==1.6.0\
tensorflow==2.3.1

## Usage Example

This is an example of how to run the NNTF with LeNet arquitecture and MNIST dataset.

```python
from NNTF import NNTF

res = NNTF(model_type='lenet', dataset_name='mnist', learning_rate=0.001, momentum=0.9, weight_decay=0.005,
                 max_epochs=30)
res.print_rqa()
res.plot_figures()
```
If you want to include a new dataset, you can check and modify the file utils/datasets.py and, then, the \_\_init\_\_ function in NNTF.py file.\
If you want to include a new neural network arquitecture, you can check and modify the file utils/models.py and, then, the \_\_init\_\_ function in NNTF.py file.\
The file full_experiments.py is to execute the whole experiments did in [1].

## Files description

1. The file NNTF.py contains the class NNTF that executes the visualization approach, including the network training, distances between matrices and RQA analysis.
The parameters are:
   - model_type: the name of the network architecture ('lenet' or 'VGG').
   - dataset_name: the name of the dataset under analysis ('mnist' or 'cifar10').
   - learning_rate: the learning rate used in the training process (default: 0.001).
   - momentum: the momentum used in the training process (default: 0.9).
   - weight_decay: the weight decay used in the training process (default: 0.0005).
   - max_epochs: the maximum number of epochs to perform the learning (default: 100).
   - batch_size: the batch size used in the training process (default: 128).
   - k_steps: the number of epochs to store a snapshot of the training process (default: 1).
   - l_min: the minimum length used in the RQA measures (default: 2).
   - interval: the bands interval used to computes the RQA measure (default: 0.05).

2. The folder data stores the data produced by the NNTF execution.
It should include the folders:
   - dist_matrix
   - models
   - RQA_measure
   - snapshots_info
   - figures

3. The folder utils contains python files.
   - The file datasets.py has functions to get datasets from keras datasets. In this case, the datasets used are MNIST and Cifar-10.
   - The file models.py has function to build the network architecture under analysis. In this case, the architectures are LeNet-5 and VGG-16.
   - The files differences.py has functions to compute the transitions and the distance between matrices used by NNTF.
   - The file measure.py contains the class RQA that computes the laminarity and the entropy measures to conduct the RQA analysis used by NNTF.
   - The file heatmaps.py has function to plot figures of the transitions and the distance matrices.

4. The file full_experiments.py is to execute the whole experiments did in [1].

5. The file requirements contains all the libraries used to execute NNFT source code.

## Reference

[1] Ferreira, M. D., Cantareira, G. D., de Mello, R. F., & Paulovich, F. V. (2021). Neural network training fingerprint: visual analytics of the training process in classification neural networks. Journal of Visualization, 1-20.

