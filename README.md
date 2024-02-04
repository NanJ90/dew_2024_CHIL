# Preparation- installing necessary packages:
The required libraries and packages are stored in requirements.txt. run `pip install -r requirements.txt`

# The project file structure:
```
project_root/
├── src/ # Source files
│ ├── experiment.py # Main script to run the project
│ ├── other .py modules for data preparation, model training, and evaluation. 
├── data/ # Data files (not all data may be relevant to every module)
│ ├── dataset.csv # Dataset file 
│ └── unrelated_data.csv # Unrelated data file
├── Experiment_Trails/ # Results from each dataset under the dataset name
│ └── datasets
    └── MAR_timestamp
    └── MCAR_timestamp
    └── MNAR_timestamp
| └── unrelated_datasets
├── requirements.txt # Project dependencies
├── analysis
......
└── README.md # The top-level README for developers using this project
```


