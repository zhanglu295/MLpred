# MLpred

## MLpred: A machine learning framework for disease risk prediction using genotype and L&E (lifestyle&environmental) factors

## Download:
```
git clone https://github.com/zhanglu295/MLpred.git
```
To download example files, go to <a href="https://drive.google.com/file/d/1_bc3qEaujjH4RawPH8rCQ96n5HRxhNwo/view?usp=sharing">Google Drive Link</a>. 
## Third party software and packages
#### MLpred would invoke <a href="https://www.cog-genomics.org/plink/1.9/">PLink</a> to perform association analysis. Please download it first and add its path to ./bashrc

#### MLpred was written in python3 and several packages should be loaded first. We highly recommend to install <a gref="https://docs.anaconda.com/anaconda/install/">anaconda</a>, which includes all the essential packages.


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
##### A. Genotype file. MLpred requries the individual's genotype data in PLink binary format (with .bed/.bim/.fam).

```
-g1 The prefix of genotype files (without .bed) of training set
-g2 The prefix of genotype files (without .bed) of validation set
-g3 The prefix of genotype files (without .bed) of test set
```

##### B. Only used in MLpred_Joint.py. Individual's L&E factors. It should be a plain text file and each row is a L&E factor and each column represents an individual.

```
-p1 L&E file for training set
-p2 L&E file for validation set
-p3 L&E file for test set
```

#### Optional parameters:
##### C. Covariate matrix of training set(in PLink format and used for association analysis). This is an optional input containing relevant covariates for association analysis (such as gender, age, PCs from genotype data etc.). The format could be found in https://www.cog-genomics.org/plink/1.9/input#covar).

```
-covar covariates for training set
```

### Output 
#### When only genotype data are used for prediction (using MLpred_Geno.py)

MLpred_Geno.py would generate the personal risk scores and AUC scores for validation and test sets:

A. AUC scores for validation and test sets (Ensemble_Geno/AUC_Ensemble.scores).

B. Personal risk scores for validation (Ensemble_Geno/Risk_geno_valid.scores) and test sets (/Ensemble_Geno/Risk_geno_test.scores).

#### When both genotype and L&E data are used for the prediction (using MLpred_Joint.py)

MLpred_Joint.py would generate the personal risk scores and AUC scores for validation and test sets:

A. AUC scores for validation and test sets (Ensemble_All/AUC_Ensemble.scores).

B. Personal risk scores for validation (Ensemble_All/Risk_All_valid.scores) and test sets (Ensemble_All/Risk_All_test.scores).

## Key steps in MLpred

### Step 1:
#### Run association analysis on training set using logistic regression in PLink (including covariates is optional). This step generates train_assoc.logistic.adjust in the association_analysis folder.
### Step 2:
#### Select candidate SNPs. MLpred would extract the SNPs with p-values smaller than 5E-3, 5E-4, 5E-5 and 5E-6 (adjusted by genomic control) by using logistic regression (--logistic in PLink). The step generates 5E3_SNP, 5E4_SNP, 5E5_SNP, 5E6_SNP in the association_analysis folder.
### Step 3:
#### LD pruning and extract the remaining SNPs from train, validation and test datasets (--indep-pairwise 50 5 0.5 in PLink). This step generates .raw files in the LD_pruning folder. 
### Step 4:
#### Recode genotype and L&E into .npy files. This step converts .raw and L&E files into .npy files and allocates them in train, valid and test folders, respectively.
### Step 5:
#### Generate candidate models from Neural Network (NN), adaboost (Ada),Gradient Boosting (GB), Lasso Regression (LR), Random Forest (RF) in Ensemble_Geno (for genotype data) and Ensemble_LE (for L&E data).
### Step 6:
#### Calculate personal risk scores and AUC scores by ensemble learning. This step generates final outputs in Ensemble_Geno and Ensemble_All, respectively.


## Troubleshooting:
### Please submit issues on this github page.
###Or contact with me through ericluzhang@comp.hkbu.edu.hk






