# IBNet: Optimizing Feature Interaction via Information Bottleneck for CTR Prediction
## Model Overview
![image](https://github.com/Acer888/IBNet/assets/45092309/2ea69736-536a-424b-a7e5-acf0a773b989)


## Requirements
python==3.8.18  
torch==1.12.0+cu116  
fuxictr  

### Others
```python
pip install -r requirement.txt
```

## Experiment results
![image](https://github.com/Acer888/IBNet/assets/45092309/dc701693-b95f-4387-a202-d862a64a2585)


## Datasets
Get the datasets from [https://github.com/openbenchmark/BARS/tree/main/datasets](https://github.com/reczoo/Datasets?tab=readme-ov-file#ctr-prediction)



## Hyperparameter settings and logs

Get the result from https://github.com/Acer888/IBNet/tree/master/IBNet_torch/checkpoints

Get Mice(PBMish) activation function from https://github.com/Acer888/IBNet/blob/master/fuxictr/pytorch/layers/activations.py




## Acknowledgement
This implementation is based on FuxiCTR and BARS. Thanks for their sharing and contribution.  
BARS: https://github.com/openbenchmark  
FuxiCTR: https://github.com/xue-pai/FuxiCTR
