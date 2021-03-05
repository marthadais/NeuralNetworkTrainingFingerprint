# Neural Network Training Fingerprint
Neural Network Training Fingerprint (NNTF) is a visualization approach to analyze the training process of any neural network performing classification [1].

## Files description
1. The file datasets.py has functions to get datasets from keras datasets. In this case, the datasets used are MNIST and Cifar-10.
2. The file models.py has function to build the network architecture under analysis. In this case, the architectures are LeNet-5 and VGG-16.
3. The files differences.py has functions to compute the transitions and the distance between matrices used by NNTF.
4. The file measure.py contains the class RQA that computes the laminarity and the entropy measures to conduct the RQA analysis used by NNTF.
5. The file heatmaps.py has function to plot figures of the transitions and the distance matrices.

## Reference

[1]

