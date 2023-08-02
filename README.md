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
@misc{deng2023learning,
      title={Learning Structured Components: Towards Modular and Interpretable Multivariate Time Series Forecasting}, 
      author={Jinliang Deng and Xiusi Chen and Renhe Jiang and Du Yin and Yi Yang and Xuan Song and Ivor W. Tsang},
      year={2023},
      eprint={2305.13036},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
