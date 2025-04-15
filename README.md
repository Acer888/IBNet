# IBNet: Optimizing Feature Interaction via Information Bottleneck for CTR Prediction
## Model Overview
![1](https://github.com/user-attachments/assets/389691b7-b8de-49a7-993a-c354577ecec8)





## Requirements
python==3.8.18  
torch==1.12.0+cu116  
fuxictr  

### Others
```python
pip install -r requirement.txt
```

## Experiment results
![2](https://github.com/user-attachments/assets/c59e9bef-ef3a-43a5-ad8f-8977b6462378)




## Datasets
Get the datasets from [https://github.com/openbenchmark/BARS/tree/main/datasets](https://github.com/reczoo/Datasets?tab=readme-ov-file#ctr-prediction)



## Hyperparameter settings and logs

Get the result from https://github.com/Acer888/IBNet/tree/master/IBNet_torch/checkpoints

Get Mice(PBMish) activation function from https://github.com/Acer888/IBNet/blob/master/fuxictr/pytorch/layers/activations.py




## Acknowledgement
This implementation is based on FuxiCTR and BARS. Thanks for their sharing and contribution.  
BARS: https://github.com/openbenchmark  
FuxiCTR: https://github.com/xue-pai/FuxiCTR
