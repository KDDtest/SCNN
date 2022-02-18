# NormNet
This is an implementation of NormNet. The skeleton of the code refers to [ST-Norm](https://github.com/JLDeng/ST-Norm).

# Requirements
Python 3.7  
Numpy >= 1.17.4  
Pandas >= 1.0.3  
Pytorch >= 1.4.0 
h5py

 
## Model Training
```
python main.py -mode train -normA 1 -normB 1 -normC 1 -normD 1 0
```
### Arguments 
normA: whether use the normalization block (a) introduced in the paper.  
normB: whether use the normalization block (b) introduced in the paper.  
normC: whether use the normalization block (c) introduced in the paper.   
normD: whether use the normalization block (d) introduced in the paper.   
dataset: dataset name.  
version: version number.  
hidden_channels: number of hidden channels.  
n_pred: number of output steps.  
n_his: number of input steps.  
n_layers: number of hidden layers.

## Model Evaluation
```
python main.py -mode eval -normA 1 -normB 1 -normC 1 -normD 1 0
```
