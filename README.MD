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

Small CharCNN performance:
```
313/313 [==============================] - 2538s 8s/step - loss: 0.0386 - accuracy: 0.9873 - val_loss: 1.6373e-05 - val_accuracy: 1.0000
```

Bidirectional LSTM + Dense performance
```
313/313 [==============================] - 4209s 13s/step - loss: 0.5117 - acc: 0.7398 - val_loss: 0.3425 - val_acc: 0.8650
```

Above, they are results of training by CharCNN model vs our custom Bidirectional LSTM in only 1 epochs on IMDB Dataset of 50K Movie Reviews. Although when we train to 3 epochs Bi-LSTM model can reach to 97 accuracy in validation data, CharCNN is the model reach to that result faster.


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


