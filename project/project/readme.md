# PREDICTION OF PROTEIN SUBCELLULAR LOCALIZATION BASED ON MICROSCOPICIMAGES

Source code for CS385 project


# Prequisites

```
torch==1.2.0
numpy==1.16.4
pandas==0.24.2
imbalanced_learn==0.5.0
imblearn==0.0
scikit_learn==0.21.3
PyYAML==5.1.2
```

# Running the code

The main script of this repo is `main.py` in each folder.

Most of the configuration is written in `config/config.yaml` in each folder

* `./code`: contains code for single-instance and online method
* `./end2end`: contains code for offline method
* `./k_fold`: contains code for k fold cross validation
