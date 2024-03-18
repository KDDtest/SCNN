# Structured Component-based Neural Network
This is an implementation of SCNN.

# Requirements
Python 3.7  
numpy >= 1.18.5  
pandas >= 1.0.3  
torch >= 1.12.0  
pytorchts == 0.6.0  
h5py  
[pytorchts](https://github.com/zalandoresearch/pytorch-ts)

 
## Model Training
```
python -u main.py -mode train -short_term 8 -long_term 144 -n_local_input 2 -cuda 0
```
### Arguments 
short_term: length of short term.  
long_term: length of long term.  
dataset: dataset name.  
version: version number.  
hidden_channels: number of hidden channels.  
n_pred: number of output steps.  
n_his: number of input steps.  
n_local_input: kernel size of causual convolution.  
n_layers: number of hidden layers.  
cuda: cuda device id.  


## Model Evaluation
```
python -u main.py -mode eval -cuda 0
```

## Citation
```
@article{deng2024disentangling,
  title={Disentangling Structured Components: Towards Adaptive, Interpretable and Scalable Time Series Forecasting},
  author={Deng, Jinliang and Chen, Xiusi and Jiang, Renhe and Yin, Du and Yang, Yi and Song, Xuan and Tsang, Ivor W},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```
