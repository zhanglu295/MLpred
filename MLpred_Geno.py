#!/usr/bin/env python

############################################################################
# Copyright (c) 2019 Hong Kong Baptist University
# Copyright (c) 2016-2018 Stanford University
# All Rights Reserved
# See file LICENSE for details.
# Author: Eric Lu Zhang
# Email: ericluzhang@comp.hkbu.edu.hk
############################################################################

## MLPred_Geno: Disease risk prediction based on ensemble machine learning model on genotype data

### Inputs：
# Required:
# A. Genotype file. MLpred_Geno requries the individual's genotype data in Plink binary format (with .bed/.bim/.fam).
# 1. The prefix of genotype files (without .bed) of training set
# 2. The prefix of genotype files (without .bed) of validation set
# 3. The prefix of genotype files (without .bed) of test set
# Option:
# C. Covariate matrix of training set(in Plink format used for association analysis)
# This is an option input containing the relevant covariates for association analysis (such as gender, age, PCs from genotype data etc.). The format could be found in https://www.cog-genomics.org/plink/1.9/input#covar).
### Outputs: 
# We will generate the personal risk scores and AUC scores for validation and test sets, we have:
# A. AUC scores in validation and test sets (./Ensemble_Geno/AUC_Ensemble.scores).
# B. Personal risk scores in validation (./Ensemble_Geno/Risk_geno_valid.scores) and test sets (./Ensemble_Geno/Risk_geno_test.scores).
#
# 
# 
#%%
import sys
import os
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib
import re
import numpy as np
import operator
from scipy import stats as sta
import math
import scipy
from itertools import groupby
from operator import itemgetter
from scipy.interpolate import spline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import pickle
from  sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import argparse
import warnings
warnings.filterwarnings("ignore")
parser = ArgumentParser(description="MLPred_Geno is an ensemble learning model to predict disease using genetic markers",add_help=False)

parser.add_argument('-v', '--version', action='version',
                    version='%(prog)s 1.0', help="Show program's version message.")
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Show this help message.')

parser.add_argument('--geno_train','-g1',type=str,help="Genotype of training set(prefix of .bed file in Plink format).", default='None')
parser.add_argument('--geno_valid','-g2',type=str,help="Genotype of validation set(prefix of .bed file in Plink format).", default='None')
parser.add_argument('--geno_test','-g3', type=str,help="Genotype of training set(prefix of .bed file in Plink format).", default='None')
parser.add_argument('--covariates','-covar', type=str,help="File with covariates for training set in Plink format.",default='None')
parser.add_argument('--outpath','-out', type=str,help="Output path (default: current folder).",default='./')
args = parser.parse_args()

def SNP_selection(args):
    infile=open(args.outpath+'/association_analysis/train_assoc.assoc.logistic.adjusted')
    snplist1=open(args.outpath+'/association_analysis/5E3_SNP','w')
    snplist2=open(args.outpath+'/association_analysis/5E4_SNP','w')
    snplist3=open(args.outpath+'/association_analysis/5E5_SNP','w')
    snplist4=open(args.outpath+'/association_analysis/5E6_SNP','w')
    SNP1=0
    SNP2=0
    SNP3=0
    SNP4=0
    linenum=0

    for line in infile:
        if linenum>0:
            A=line.strip('\n').rsplit()
            if float(A[3])<0.005:
                snplist1.write(A[1]+'\n')
                SNP1+=1
            if float(A[3])<0.0005:
                snplist2.write(A[1]+'\n')
                SNP2+=1
            if float(A[3])<0.00005:
                snplist3.write(A[1]+'\n')
                SNP3+=1
            if float(A[3])<0.000005:
                snplist4.write(A[1]+'\n')
                SNP4+=1
        linenum+=1
    infile.close()
    snplist1.close()
    snplist2.close()
    snplist3.close()
    snplist4.close()

def LD_pruning(args):
    if not os.path.exists(args.outpath+'/LD_pruning'):
        cmd='mkdir '+args.outpath+'/LD_pruning'
        os.system(cmd)
    para=args.geno_train
    if os.stat(args.outpath+"/association_analysis/5E3_SNP").st_size!=0:
        os.system('plink --bfile '+para+' --extract '+args.outpath+'/association_analysis/5E3_SNP --make-bed --out '+args.outpath+'/LD_pruning/5E3_train_extract')
        if len(open(args.outpath+"/association_analysis/5E3_SNP",'rU').readlines())!=1:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E3_train_extract --indep-pairwise 50 5 0.5 --out '+args.outpath+'/LD_pruning/train_5E3')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E3_train_extract --prune --extract '+args.outpath+'/LD_pruning/train_5E3.prune.in --make-bed --out '+args.outpath+'/LD_pruning/train_5E3_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/train_5E3_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E3_recodeA')
        else:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E3_train_extract --prune --make-bed --out '+args.outpath+'/LD_pruning/train_5E3_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E3_train_extract --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E3_recodeA')

    if os.stat(args.outpath+"/association_analysis/5E4_SNP").st_size!=0:
        os.system('plink --bfile '+para+' --extract '+args.outpath+'/association_analysis/5E4_SNP --make-bed --out '+args.outpath+'/LD_pruning/5E4_train_extract')
        if len(open(args.outpath+"/association_analysis/5E4_SNP",'rU').readlines())!=1:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E4_train_extract --indep-pairwise 50 5 0.5 --out '+args.outpath+'/LD_pruning/train_5E4')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E4_train_extract --prune --extract '+args.outpath+'/LD_pruning/train_5E4.prune.in --make-bed --out '+args.outpath+'/LD_pruning/train_5E4_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/train_5E4_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E4_recodeA')
        else:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E4_train_extract --prune --make-bed --out '+args.outpath+'/LD_pruning/train_5E4_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E4_train_extract --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E4_recodeA')

    if os.stat(args.outpath+"/association_analysis/5E5_SNP").st_size!=0:
        os.system('plink --bfile '+para+' --extract '+args.outpath+'/association_analysis/5E5_SNP --make-bed --out '+args.outpath+'/LD_pruning/5E5_train_extract')
        if len(open(args.outpath+"/association_analysis/5E5_SNP",'rU').readlines())!=1:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E5_train_extract --indep-pairwise 50 5 0.5 --out '+args.outpath+'/LD_pruning/train_5E5')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E5_train_extract --prune --extract '+args.outpath+'/LD_pruning/train_5E5.prune.in --make-bed --out '+args.outpath+'/LD_pruning/train_5E5_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/train_5E5_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E5_recodeA')
        else:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E5_train_extract --prune --make-bed --out '+args.outpath+'/LD_pruning/train_5E5_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E5_train_extract --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E5_recodeA')
    if os.stat(args.outpath+"/association_analysis/5E6_SNP").st_size!=0:
        os.system('plink --bfile '+para+' --extract '+args.outpath+'/association_analysis/5E6_SNP --make-bed --out '+args.outpath+'/LD_pruning/5E6_train_extract')
        if len(open(args.outpath+"/association_analysis/5E6_SNP",'rU').readlines())!=1:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E6_train_extract --indep-pairwise 50 5 0.5 --out '+args.outpath+'/LD_pruning/train_5E6')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E6_train_extract --prune --extract '+args.outpath+'/LD_pruning/train_5E6.prune.in --make-bed --out '+args.outpath+'/LD_pruning/train_5E6_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/train_5E6_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E6_recodeA')
        else:
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E6_train_extract --prune --make-bed --out '+args.outpath+'/LD_pruning/train_5E6_pruning')
            os.system('plink --bfile '+args.outpath+'/LD_pruning/5E6_train_extract --prune --recode A --out '+args.outpath+'/LD_pruning/train_5E6_recodeA')

    for data in ['valid','test']:
        para=''
        if data=='valid':
            para=args.geno_valid
        if data=='test':
            para=args.geno_test
        if os.stat(args.outpath+"/association_analysis/5E3_SNP").st_size!=0:
            if len(open(args.outpath+"/association_analysis/5E3_SNP",'rU').readlines())!=1:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/LD_pruning/train_5E3.prune.in --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E3_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E3_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E3_recodeA')
            else:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/association_analysis/5E3_SNP --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E3_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E3_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E3_recodeA')

        if os.stat(args.outpath+"/association_analysis/5E4_SNP").st_size!=0:
            if len(open(args.outpath+"/association_analysis/5E4_SNP",'rU').readlines())!=1:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/LD_pruning/train_5E4.prune.in --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E4_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E4_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E4_recodeA')
            else:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/association_analysis/5E4_SNP --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E4_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E4_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E4_recodeA')
        if os.stat(args.outpath+"/association_analysis/5E5_SNP").st_size!=0:
            if len(open(args.outpath+"/association_analysis/5E5_SNP",'rU').readlines())!=1:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/LD_pruning/train_5E5.prune.in --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E5_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E5_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E5_recodeA')
            else:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/association_analysis/5E5_SNP --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E5_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E5_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E5_recodeA')
        if os.stat(args.outpath+"/association_analysis/5E6_SNP").st_size!=0:
            if len(open(args.outpath+"/association_analysis/5E6_SNP",'rU').readlines())!=1:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/LD_pruning/train_5E6.prune.in --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E6_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E6_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E6_recodeA')
            else:
                os.system('plink --bfile '+para+' --prune --extract '+args.outpath+'/association_analysis/5E6_SNP --make-bed --out '+args.outpath+'/LD_pruning/'+data+'_5E6_pruning')
                os.system('plink --bfile '+args.outpath+'/LD_pruning/'+data+'_5E6_pruning --prune --recode A --out '+args.outpath+'/LD_pruning/'+data+'_5E6_recodeA')

def generate_G_P(path_diseaseid,path_data):
    phenotype=np.genfromtxt(path_diseaseid,dtype=str)
    data=np.genfromtxt(path_data,dtype=str)
    SNPindexlist=[]
    SNP_id=[]
    ind_id=[]
    indindexlist=[]
    ind_genotype=[]
    all_disease=[]
    totalind=phenotype.shape[0]
    for i in range(totalind+1):
        if i==0:
            snpnum=data[0].shape[0]
            for j in range(6,snpnum):
                SNPindexlist.append(j)
                SNP_id.append(data[0][j].split('_')[0])
        else:
            if len(SNPindexlist)==1:
                tmp=data[i][6]
                if tmp=='NA':
                    tmp='0'
            else:
                tmp=list(itemgetter(*SNPindexlist)(data[i]))
                for x in range(len(tmp)):
                    if tmp[x]=='NA':
                        tmp[x]='0'
            indindexlist.append(i-1)
            ind_genotype.append(list(map(int,tmp)))
            ind_id.append(phenotype[i-1][0])
            all_disease.append(int(data[i][5])-1)
    all_genotype=sta.zscore(ind_genotype)
    s=np.isnan(all_genotype)
    all_genotype[s]=1
    return all_genotype,all_disease,ind_id,SNP_id


def transfer_plain_npy(args):
    if not os.path.exists(args.outpath+'/train'):
        os.system('mkdir '+args.outpath+'/train')
    if not os.path.exists(args.outpath+'/valid'):
        os.system('mkdir '+args.outpath+'/valid')
    if not os.path.exists(args.outpath+'/test'):
        os.system('mkdir '+args.outpath+'/test')
    outpath1=args.outpath+'/train/'
    outpath2=args.outpath+'/valid/'
    outpath3=args.outpath+'/test/'
    for i in ['5E3','5E4','5E5','5E6']:
        if os.path.exists(args.outpath+'/LD_pruning/train_'+i+'_recodeA.raw')==False:
            continue
        out_train_disease=outpath1+'disease_train'
        out_valid_disease=outpath2+'disease_valid'
        out_test_disease=outpath3+'disease_test'

        out_geno_train=outpath1+'genotype_train_'+i
        out_geno_valid=outpath2+'genotype_valid_'+i
        out_geno_test=outpath3+'genotype_test_'+i

        out_train_id=outpath1+'indid_train'
        out_valid_id=outpath2+'indid_valid'
        out_test_id=outpath3+'indid_test'

        out_train_pheno=outpath1+'phenotype_train'
        out_valid_pheno=outpath2+'phenotype_valid'
        out_test_pheno=outpath3+'phenotype_test'

        [train_genotype,train_disease,train_indid,train_SNPid]=generate_G_P(args.outpath+'/LD_pruning/train_'+i+'_pruning.fam',args.outpath+'/LD_pruning/train_'+i+'_recodeA.raw')
        [valid_genotype,valid_disease,valid_indid,valid_SNPid]=generate_G_P(args.outpath+'/LD_pruning/valid_'+i+'_pruning.fam',args.outpath+'/LD_pruning/valid_'+i+'_recodeA.raw')
        [test_genotype,test_disease,test_indid,test_SNPid]=generate_G_P(args.outpath+'/LD_pruning/test_'+i+'_pruning.fam',args.outpath+'/LD_pruning/test_'+i+'_recodeA.raw')

        np.save(out_geno_train,train_genotype)
        np.save(out_geno_valid,valid_genotype)
        np.save(out_geno_test,test_genotype)
        np.save(out_train_id,train_indid)
        np.save(out_valid_id,valid_indid)
        np.save(out_test_id,test_indid)
        np.save(out_train_disease,train_disease)
        np.save(out_valid_disease,valid_disease)
        np.save(out_test_disease,test_disease)

def ML_model_Geno(args):
    if not os.path.exists(args.outpath+'/ML_Geno'):
        os.system('mkdir '+args.outpath+'/ML_Geno')
    disease_train=np.load(args.outpath+'/train/disease_train.npy')
    Cxlist_NN=[(20,20,20),(30,30),(10,10,10,10),(30,20,10)]
    Cxlist_ada=[DecisionTreeClassifier(),LogisticRegression(),ExtraTreeClassifier(),GaussianNB()]
    Cxlist_GB=[0.0001,0.001,0.01,0.1]
    Cxlist_LR=[0.0001,0.001,0.01,0.1]
    Cxlist_RF=[0.0001,0.001,0.01,0.1]
    slist=[]
    if os.path.exists(args.outpath+'/train/genotype_train_5E3.npy'):
        slist.append(1)
    if os.path.exists(args.outpath+'/train/genotype_train_5E4.npy'):
        slist.append(2)
    if os.path.exists(args.outpath+'/train/genotype_train_5E5.npy'):
        slist.append(3)
    if os.path.exists(args.outpath+'/train/genotype_train_5E6.npy'):
        slist.append(4)

    for s in slist:
        if s==1:
            Geno_train=np.load(args.outpath+'/train/genotype_train_5E3.npy')
            Geno_valid=np.load(args.outpath+'/valid/genotype_valid_5E3.npy')
            Geno_test=np.load(args.outpath+'/test/genotype_test_5E3.npy')
        elif s==2:
            Geno_train=np.load(args.outpath+'/train/genotype_train_5E4.npy')
            Geno_valid=np.load(args.outpath+'/valid/genotype_valid_5E4.npy')
            Geno_test=np.load(args.outpath+'/test/genotype_test_5E4.npy')
        elif s==3:
            Geno_train=np.load(args.outpath+'/train/genotype_train_5E5.npy')
            Geno_valid=np.load(args.outpath+'/valid/genotype_valid_5E5.npy')
            Geno_test=np.load(args.outpath+'/test/genotype_test_5E5.npy')
        else:
            Geno_train=np.load(args.outpath+'/train/genotype_train_5E6.npy')
            Geno_valid=np.load(args.outpath+'/valid/genotype_valid_5E6.npy')
            Geno_test=np.load(args.outpath+'/test/genotype_test_5E6.npy')

        for t in [1,2,3,4]:
            Cx_NN=Cxlist_NN[t-1]
            NN=MLPClassifier(hidden_layer_sizes=Cx_NN,max_iter=1000)
            NN.fit(Geno_train,disease_train)
            Y=NN.predict_proba(Geno_valid)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_NN_valid.npy',Y[:,1])
            Y=NN.predict_proba(Geno_test)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_NN_test.npy',Y[:,1])

            Cx_ada=Cxlist_ada[t-1]
            ada = AdaBoostClassifier(base_estimator=Cx_ada)
            ada.fit(Geno_train,disease_train)
            Y=ada.predict_proba(Geno_valid)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_ada_valid.npy',Y[:,1])
            Y=ada.predict_proba(Geno_test)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_ada_test.npy',Y[:,1])

            Cx_GB=Cxlist_GB[t-1]
            GB = GradientBoostingClassifier(min_impurity_decrease=Cx_GB)
            GB.fit(Geno_train,disease_train)
            Y=GB.predict_proba(Geno_valid)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_GB_valid.npy',Y[:,1])
            Y=GB.predict_proba(Geno_test)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_GB_test.npy',Y[:,1])

            Cx_LR=Cxlist_LR[t-1]
            LR = LogisticRegression(penalty='l1', C=Cx_LR, max_iter=10000)
            LR.fit(Geno_train,disease_train)
            Y=LR.predict_proba(Geno_valid)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_LR_valid.npy',Y[:,1])
            Y=LR.predict_proba(Geno_test)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_LR_test.npy',Y[:,1])

            Cx_RF=Cxlist_RF[t-1]
            RF = RandomForestClassifier(min_impurity_decrease=Cx_RF)
            RF.fit(Geno_train,disease_train)
            Y=RF.predict_proba(Geno_valid)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_RF_valid.npy',Y[:,1])
            Y=RF.predict_proba(Geno_test)
            np.save(args.outpath+'/ML_Geno/'+str(s)+'_'+str(t)+'_RF_test.npy',Y[:,1])
def ensemble_model(args):
    disease_valid=np.load(args.outpath+'/valid/disease_valid.npy')
    disease_test=np.load(args.outpath+'/test/disease_test.npy')
    slist=[]
    if os.path.exists(args.outpath+'/train/genotype_train_5E3.npy'):
        slist.append('1')
    if os.path.exists(args.outpath+'/train/genotype_train_5E4.npy'):
        slist.append('2')
    if os.path.exists(args.outpath+'/train/genotype_train_5E5.npy'):
        slist.append('3')
    if os.path.exists(args.outpath+'/train/genotype_train_5E6.npy'):
        slist.append('4')
    for i in ['GB','NN','ada','RF','LR']:
        if i=='GB':
            filename1=args.outpath+'/ML_Geno/1_1_'+i+'_valid.npy'
            filename2=args.outpath+'/ML_Geno/1_1_'+i+'_test.npy'
            data1=np.load(filename1)
            data2=np.load(filename2)
            enseinput=data1.reshape(-1,1)
            ensetest=data2.reshape(-1,1)
            for m in slist:
                for n in ['1','2','3','4']:
                    filename1=args.outpath+'/ML_Geno/'+m+'_'+n+'_'+i+'_valid.npy'
                    filename2=args.outpath+'/ML_Geno/'+m+'_'+n+'_'+i+'_test.npy'
                    data1=np.load(filename1)
                    data2=np.load(filename2)
                    if i=='GB' and m=='1' and n=='1':
                        continue
                    enseinput=np.concatenate((enseinput,data1.reshape(-1,1)),axis=1)
                    ensetest=np.concatenate((ensetest,data2.reshape(-1,1)),axis=1)
    LR = LogisticRegression(penalty='l1', max_iter=10000,C=10000)
    LR.fit(enseinput,disease_valid)
    Y1=LR.predict_proba(enseinput)
    Y2=LR.predict_proba(ensetest)
    if not os.path.exists(args.outpath+'/Ensemble_Geno'):
        os.system('mkdir '+args.outpath+'/Ensemble_Geno')

    out1=open(args.outpath+'/Ensemble_Geno/Risk_geno_valid.scores','w')
    out2=open(args.outpath+'/Ensemble_Geno/Risk_geno_test.scores','w')
    for x in Y1[:,1]:
        out1.write(str(round(x,3))+'\n')
    out1.close()
    for x in Y2[:,1]:
        out2.write(str(round(x,3))+'\n')
    out2.close()

    out=open(args.outpath+'/Ensemble_Geno/AUC_Ensemble.scores','w')
    AUC_valid=roc_auc_score(disease_valid,Y1[:,1])
    AUC_test=roc_auc_score(disease_test,Y2[:,1])
    out.write('AUC score for valid set '+str(round(AUC_valid,3))+'\n')
    out.write('AUC score for test set '+str(round(AUC_test,3))+'\n')
    out.close()
if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.stdout.write("Please use 'python MLpred_Geno.py -h' for more information!\n")
    elif args.geno_train=="None" or args.geno_valid=="None" or args.geno_test=="None":
        if args.geno_train=="None":
            print('Please provide genotype of training set!\n')
        if args.geno_valid=="None":
            print('Please provide genotype of validation set!\n')
        if args.geno_test=="None":
            print('Please provide genotype of test set!\n')
    else:
        valid=1
        if not os.path.exists(args.geno_train+'.bed'):
            print('Genotype file of training set does not exist!\n')
            valid=0
        if not os.path.exists(args.geno_valid+'.bed'):
            print('Genotype file of validation set does not exist!\n')
            valid=0
        if not os.path.exists(args.geno_test+'.bed'):
            print('Genotype file of test set does not exist!\n')
            valid=0
        if args.covariates!='None':
            if not os.path.exists(args.covariates):
                valid=0
                print('Covariate file does not exist!')
        if valid==1:
            step1_finish=1
            step2_finish=1
            step3_finish=1
            step4_finish=1
            step5_finish=1
            print('Step1: Running association analysis on training set\n')
            if not os.path.exists(args.outpath):
                os.system('mkdir '+args.outpath)
            if not os.path.exists(args.outpath+'/association_analysis'):
                cmd='mkdir '+args.outpath+'/association_analysis'
                os.system(cmd)
            if args.covariates=="None":
                cmd="plink --adjust --bfile "+args.geno_train+" --logistic --out "+args.outpath+"/association_analysis/train_assoc"
            else:
                cmd="plink --adjust --bfile "+args.geno_train+" --covar "+args.covariates+" --logistic --out "+args.outpath+"/association_analysis/train_assoc"
            os.system(cmd)
            if not os.path.exists(args.outpath+'/association_analysis/train_assoc.assoc.logistic.adjusted'):
                step1_finish=0
                step2_finish=0
                step3_finish=0
                step4_finish=0
                step5_finish=0
                print('Failed to generate train_assoc.assoc.logistic.adjusted in Step1')
            if step1_finish==1:
                print('Step1 finished\n')
                print('Step2: Selecting candidate SNPs\n')
                SNP_selection(args)
      
            if not os.path.exists(args.outpath+'/association_analysis/5E3_SNP'):
                step2_finish=0
                step3_finish=0
                step4_finish=0
                step5_finish=0
                print('Failed to generate *_SNP in Step2')
            if step2_finish==1:
                print('Step2 finished\n')
                print('Step3: LD pruning and extract the remaining SNPs from train, validation and test datasets\n')
                LD_pruning(args)
           
            if not os.path.exists(args.outpath+'/LD_pruning/train_5E3_recodeA.raw'):
                step3_finish=0
                step4_finish=0
                step5_finish=0
                print('Failed to generate *.raw in Step3')
            if step3_finish==1:
               print('Step3 finished\n')
               print('Step4: Recode genotype into .npy files')
               transfer_plain_npy(args)
            step4_finish=1
            if not os.path.exists(args.outpath+'/train/disease_train.npy') or not os.path.exists(args.outpath+'/valid/disease_valid.npy') or not os.path.exists(args.outpath+'/test/disease_test.npy'):
                step4_finish=0
                step5_finish=0
                print('Failed to generate *.npy file of genotype in Step4')
            if step4_finish==1:
               print('Step4 finished\n')
               print('Step5: Generate candidate models from Neural Network (NN), adaboost (Ada),Gradient Boosting (GB), Lasso Regression (LR), Random Forest (RF)')
               ML_model_Geno(args)
            step5_finish=1
            if not os.path.exists(args.outpath+'/ML_Geno/1_1_ada_test.npy') or not os.path.exists(args.outpath+'/ML_Geno/1_1_GB_test.npy') or not os.path.exists(args.outpath+'/ML_Geno/1_1_LR_test.npy') or not os.path.exists(args.outpath+'/ML_Geno/1_1_NN_test.npy') or not os.path.exists(args.outpath+'/ML_Geno/1_1_RF_test.npy'):
                step5_finish=0
                print('Failed to generate *.npy for individual ML model prediction in Step5')
            if step5_finish==1:
                print('Step5 finished\n')
                print('Step 6: Calculate personal risk scores and AUC scores by ensemble learning')
                ensemble_model(args)
                print('Step6 finished\n')
