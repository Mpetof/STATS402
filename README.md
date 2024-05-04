# STATS402
This is a project about protein interface prediction.

Download both the training and testing datasets from the following links:
[Train Data](https://zenodo.org/records/1127774/files/train.cpkl.gz?download=1); 
[Test Data](https://zenodo.org/records/1127774/files/test.cpkl.gz?download=1)

For a detailed description of the dataset, refer to:
[Data_Description](https://zenodo.org/records/1127774#.WkLewGGnGcY])

Please follow the following directory structure:
```
Main/
├── dataset/
│   ├── train.cpkl.gz
│   └── test.cpkl.gz
├── constructed_graphs/
├── models/
├── results/
├── Graph_Construction.py
├── inference.py
├── MLP.py
├── model_construction.py
└── training.py
```

The Graph_Construction.py script allows users to construct a protein graph from raw data, which serves as the input for our model. Users could run the script with default settings, by executing the following command:

```python
python Graph_Construction.py
```

The model_construction.py script contains all models presented in our paper, including, naive GCN, naive GAT, s1-GCN, s2-GCN, s1-GAT, s2-GAT. Models are used in the train.py script.

The train.py script allows users to train a model defined in model_construction.py file. which serves as the input for our model. To choose different specific hyperparameter settings, users need to carefully revise the script. Although we provide some options for training for users to select (after running the script, the program will ask users for specific input, like learning rate, epoch, etc.), some hyperparameters like the weight for loss function need to be carefully revised in the script. Users could run the script with the following command:

```python
python train.py
```

The MLP.py script allows users to directly train a MLP model. The parameter settings need to be revised in the script, but users could still run the default hyperparameters that we selected:

```python
python MLP.py
```

Finally, we provide an "inference.py" file that allows users to run a trained model on the test set. Simply run the script, and program will automatically asks users to select the model and provide the corresponding files needed:

```python
python inference.py
```
