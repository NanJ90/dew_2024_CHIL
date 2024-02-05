# Preparation- installing necessary packages:
The required libraries and packages are stored in requirements.txt. run `pip install -r requirements.txt`

# The project file structure:
```
project_root/
├── src/ # Source files
│ ├── main.py # Main script to run the project
│ ├── other .py modules for data preparation, model training, and evaluation. 
├── data/ # Data files (not all data may be relevant to every module)
│ ├── dataset.csv # Dataset file 
│ └── unrelated_data.csv # Unrelated data file
├── Experiment_Trails/ # Results from each dataset under the dataset name
│ └── dataset1
    └── MAR_timestamp
        └──MAR
            └──metrics and results
    └── MCAR_timestamp
        └──MCAR
    └── MNAR_timestamp
        └──MNAR
  └──dataset2
  ......
| └── unrelated_datasets
├── requirements.txt # Project dependencies
├── analysis
......
└── README.md # The top-level README for developers using this project
```
# To run the program
1. `cd src`
2. `python main.py`

The default running dataset is eeg-eye-state, and you can choose different dataset a variable called `CURRENT_SUPPORTED_DATALOADERS` inside `main.py`
