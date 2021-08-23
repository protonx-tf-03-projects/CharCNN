# CharCNN 

Library Logo <--- **FIXME**

<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

Our porject is using TensorFlow framework to implement CharCNN model to classify English sentences. 

### Model Architecture

![](image/CharCNN-architecture2.png)

### Team
- Team leader:
    1. Github: hoangcaobao
- Team members:
    1. Github: Nguyendat-bit
    2. Github: aestheteeism
- Advisor:
    1. Github: bangoc123

## I.  Set up environment
- Step 1: <--- **FIXME**

```python
conda env create -f environment.yml
```

- Step 2: <--- **FIXME**
```

```

- Step 3: <--- **FIXMEE**

```

``` 

## II.  Set up your dataset

- We use data IMBD Dataset of 50K Movie Reviews for training that can be easily found at [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- After that, dowload it locally then move this csv file to CharCNN folder without changing name
- References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer)

## III. Training Process

Training script:

```python

python train.py --epochs ${epochs} 

```
For example: 
```python

python train.py --epochs 3 

```

Due to the large amount of dataset, we recommend using epochs smaller than 3


## IV. Predict Process

You create file csv with 1 column named sentences contain your dataset 

Predict script: 
```bash
python predict.py --test-file ${link_to_test_data} --result-file ${name_of_file_contain_result}
```
For example:
```bash
python predict.py --test-file test.csv --result-file result.csv
```
## V. Result and Comparision

**FIXME**

Your implementation
```
Epoch 7/10
782/782 [==============================] - 261s 334ms/step - loss: 0.8315 - acc: 0.8565 - val_loss: 0.8357 - val_acc: 0.7978
Epoch 8/10
782/782 [==============================] - 261s 334ms/step - loss: 0.3182 - acc: 0.8930 - val_loss: 0.6161 - val_acc: 0.8047
Epoch 9/10
782/782 [==============================] - 261s 333ms/step - loss: 1.1965 - acc: 0.8946 - val_loss: 3.9842 - val_acc: 0.7855
Epoch 10/10
782/782 [==============================] - 261s 333ms/step - loss: 0.4717 - acc: 0.8878 - val_loss: 0.4894 - val_acc: 0.8262

```

**FIXME**

Other architecture

```
Epoch 6/10
391/391 [==============================] - 115s 292ms/step - loss: 0.1999 - acc: 0.9277 - val_loss: 0.4719 - val_acc: 0.8130
Epoch 7/10
391/391 [==============================] - 114s 291ms/step - loss: 0.1526 - acc: 0.9494 - val_loss: 0.5224 - val_acc: 0.8318
Epoch 8/10
391/391 [==============================] - 115s 293ms/step - loss: 0.1441 - acc: 0.9513 - val_loss: 0.5811 - val_acc: 0.7875
```

Your comments about these results <--- **FIXME**


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


