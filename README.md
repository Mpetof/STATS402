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

The Graph_Construction.py script allows users to construct a protein graph from raw data, which serves as the input for our model. To run the script with default settings, execute the following command:
```python
python Graph_Construction.py
```


