# IBNet: Optimizing Feature Interaction via Information Bottleneck for CTR Prediction
## Model Overview
![image](https://github.com/Acer888/IBNet/assets/45092309/0222d5d3-b685-4bb8-a1ba-3be433b0161d)


## Requirements
python==3.8.18  
torch==1.12.0+cu116  
fuxictr  

### Others
```python
pip install -r requirements.txt
```

## Experiment results
![image](https://github.com/Acer888/IBNet/assets/45092309/11020051-b902-477e-85c2-52a4cbe5e13c)


## Datasets
Get the datasets from [https://github.com/openbenchmark/BARS/tree/main/datasets](https://github.com/reczoo/Datasets?tab=readme-ov-file#ctr-prediction)



## Hyperparameter settings and logs

Get the result from https://github.com/Acer888/IBNet/tree/master/IBNet_torch/checkpoints

Get Mice activation function from https://github.com/Acer888/IBNet/blob/master/fuxictr/pytorch/layers/activations.py




## Acknowledgement
This implementation is based on FuxiCTR and BARS. Thanks for their sharing and contribution.  
BARS: https://github.com/openbenchmark  
FuxiCTR: https://github.com/xue-pai/FuxiCTR
