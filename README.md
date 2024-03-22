# IBNet: Optimizing Feature Interaction via Information Bottleneck for CTR Prediction
## Model Overview
![overview](https://github.com/Acer888/IBNet/assets/45092309/6069f05a-b227-4dee-804a-187cea4e501e)



## Requirements
python==3.8.18  
torch==1.12.0+cu116  
fuxictr  
```python
pip install -r requirements.txt
```

## Experiment results

![image](https://github.com/Acer888/IBNet/assets/45092309/0b654760-8ad5-48ed-974c-2a9fc9a9de20)


## Datasets
Get the datasets from [https://github.com/openbenchmark/BARS/tree/main/datasets](https://github.com/reczoo/Datasets?tab=readme-ov-file#ctr-prediction)



## Hyperparameter settings and logs

Get the result from https://github.com/Acer888/IBNet/tree/master/IBNet_torch/checkpoints

Get Mice activation function from https://github.com/Acer888/IBNet/blob/master/fuxictr/pytorch/layers/activations.py




## Acknowledgement
This implementation is based on FuxiCTR and BARS. Thanks for their sharing and contribution.  
BARS: https://github.com/openbenchmark  
FuxiCTR: https://github.com/xue-pai/FuxiCTR
