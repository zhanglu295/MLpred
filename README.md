# MLpred

# MLpred: A machine learning framework for disease risk prediction using genotype and L&amp;E (lifestyle&amp;environmental) factors

## Download:
```
git clone https://github.com/zhanglu295/MLpred.git
```
To download example files, go to <a href="https://drive.google.com/file/d/1_bc3qEaujjH4RawPH8rCQ96n5HRxhNwo/view?usp=sharing">Google Drive Link</a>. 
## Third party software
#### MLpred would invoke <a href="https://www.cog-genomics.org/plink/1.9/">PLink</a> to perform association analysis. Please download it first and add the path to ./bashrc
## Quick start:
#### if only genotype data are given
```
mkdir ./testgeno
tar zxvf example.tar.gz
python MLpred_Geno.py -g1 ./example/train_set -g2 ./example/valid_set -g3 ./example/test_set -covar ./example/train_cov.matrix -out ./testgeno/
```

#### if both of the genotype and L&E data are given
```
mkdir ./testjoint
tar zxvf example.tar.gz
python MLpred_Joint.py -g1 ./example/train_set -g2 ./example/valid_set -g3 ./example/test_set -p1 ./example/train_LE.matrix -p2 ./example/valid_LE.matrix -p3 ./example/test_LE.matrix -covar ./example/train_cov.matrix -out ./testjoint/
```

## Basic Usage
### Input
#### Required:
##### A. Genotype file. MLpred requries the individual's genotype data in Plink binary format (with .bed/.bim/.fam).
1. The prefix of genotype files (without .bed) of training set
2. The prefix of genotype files (without .bed) of validation set
3. The prefix of genotype files (without .bed) of test set

##### B. Only used in MLpred_Joint.py. Individual's L&E factors. It should be a plain text file and each row is a L&E factor and each column represents an individual.
1. L&E file for training set
2. L&E file for validation set
3. L&E file for test set

#### Option:
##### C. Covariate matrix of training set(in Plink format used for association analysis). This is an option input containing the relevant covariates for association analysis (such as gender, age, PCs from genotype data etc.). The format could be found in https://www.cog-genomics.org/plink/1.9/input#covar).

### Output 
#### When only genotype data are used for the prediction using MLpred_Geno.py
MLpred_Geno.py would generate the personal risk scores and AUC scores for validation and test sets:
 A. AUC scores in validation and test sets (/Ensemble_Geno/AUC_Ensemble.scores).
 B. Personal risk scores in validation (/Ensemble_Geno/Risk_geno_valid.scores) and test sets (/Ensemble_Geno/Risk_geno_test.scores).
#### When both genotype and L&E data are used for the prediction using MLpred_Joint.py
MLpred_Joint.py would generate the personal risk scores and AUC scores for validation and test sets:
A. AUC scores in validation and test sets (/Ensemble_All/AUC_Ensemble.scores).
B. Personal risk scores in validation (/Ensemble_All/Risk_All_valid.scores) and test sets (/Ensemble_All/Risk_All_test.scores).
 
