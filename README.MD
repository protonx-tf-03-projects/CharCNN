# CharCNN 

Our porject is using TensorFlow framework to implement CharCNN model to classify English sentences. 

### Model Architecture
![](image/CharCNN-architecture1.png)
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
- Step 1: Clone repo to local

```bash
git clone https://github.com/protonx-tf-03-projects/CharCNN.git
```

- Step 2: Change directory to folder
```bash
cd CharCNN
```

- Step 3: Make sure you have installed Miniconda. If not yet, see the setup document [here](https://conda.io/en/latest/user-guide/install/index.html#regular-installation).

- Step 4: Create environment
```bash
conda env create -f environment.yml
``` 

- Step 5: Activate environment
```bash
conda activate CharCNN
```

Congratulation. Now you have environment to use our project.

## II.  Set up your dataset

- We use data IMBD Dataset of 50K Movie Reviews for training that can be easily found at [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- After that, dowload it locally then move this csv file to CharCNN folder without changing name
- Dataset (csv file) has to follow this structure

| review  |   sentiment     |
|:---------:|:-------------:|
| setence |  label |
| ...           |.... |
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
Now you have sentiment of your sentences in test.csv in result.csv file

## V. Result and Comparision

Small CharCNN performance:
```
Epoch 1/3
313/313 [==============================] - 1583s 5s/step - loss: 0.6929 - accuracy: 0.5117 - val_loss: 0.6782 - val_accuracy: 0.5392
Epoch 2/3
313/313 [==============================] - 1554s 5s/step - loss: 0.3542 - accuracy: 0.8433 - val_loss: 0.2596 - val_accuracy: 0.8948
Epoch 3/3
313/313 [==============================] - 1481s 5s/step - loss: 0.1619 - accuracy: 0.9414 - val_loss: 0.2819 - val_accuracy: 0.8902
```

Bidirectional LSTM + Dense performance:
```
Epoch 1/3
313/313 [==============================] - 5085s 16s/step - loss: 0.4762 - acc: 0.7655 - val_loss: 0.4360 - val_acc: 0.8097
Epoch 2/3
313/313 [==============================] - 4386s 14s/step - loss: 0.2837 - acc: 0.8928 - val_loss: 0.2808 - val_acc: 0.8905
Epoch 3/3
313/313 [==============================] - 2185s 7s/step - loss: 0.1841 - acc: 0.9351 - val_loss: 0.3262 - val_acc: 0.8924
```

Above, they are results of training of CharCNN model vs our custom Bidirectional LSTM in 3 epochs on IMDB Dataset of 50K Movie Reviews. CharCNN is not good as Bidirectional LSTM.


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./model/tests` folder please run:

```bash
pytest
```


