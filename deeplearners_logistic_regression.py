# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:37:11 2017
@author: raoki
"""

import pandas as pd
import sklearn.linear_model as lm
import sklearn.cross_validation as cv
import sklearn.metrics as metrics
import numpy as np

'''load datasets'''
genes_da = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\features_importance_DA_all.csv')

genes_dfs1 = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\features_importance_DFS_all_os.csv')
genes_dfs2 = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\features_importance_DFS_all_er.csv')
genes_dfs3 = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\features_importance_DFS_all_her.csv')

genes_rf1 = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\RF_Optimal_Features_overall_survival.csv')
genes_rf2 = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\RF_Optimal_Features_ER_status.csv')
genes_rf3 = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\RF_Optimal_Features_HER2_status.csv')


'''Same Column Names and top 100'''
def transform(df):
    df.columns = ['genes','importance']
    df = df.sort_values('importance',ascending=False)
    df100 = df[0:100]
    return df, df100

genes_da, genes_da_100 = transform(genes_da)
genes_dfs1, genes_dfs1_100 = transform(genes_dfs1)
genes_dfs2, genes_dfs2_100 = transform(genes_dfs2)
genes_dfs3, genes_dfs3_100 = transform(genes_dfs3)
genes_rf1, genes_rf1_100 = transform(genes_rf1)
genes_rf2, genes_rf2_100 = transform(genes_rf2)
genes_rf3, genes_rf3_100 = transform(genes_rf3)


genes_da_100 = genes_da_100.reset_index(drop=True)

'''intersection between top100'''
def print_intersection(gene1,gene2,gene3,c_information):
    print('Intersection between DA and DSF - ',c_information,' - ',
          len(list(set(gene1.genes).intersection(gene2.genes))))
    print(list(set(gene1.genes).intersection(gene2.genes)))
    print('\n')
    print('Intersection between DA and RF - ',c_information,' - ',
          len(list(set(gene1.genes).intersection(gene3.genes))))
    print(list(set(gene1.genes).intersection(gene3.genes)))
    print('\n')
    print('Intersection between RF and DSF - ',c_information,' - ',
          len(list(set(gene2.genes).intersection(gene3.genes))))
    print(list(set(gene2.genes).intersection(gene3.genes)))
    print('\n')
    return [len(list(set(gene1.genes).intersection(gene2.genes))),
            len(list(set(gene1.genes).intersection(gene3.genes))),
            len(list(set(gene2.genes).intersection(gene3.genes)))]

int1 = print_intersection(genes_da_100,genes_dfs1_100,genes_rf1_100,'Overall Survival')
int2 = print_intersection(genes_da_100,genes_dfs2_100,genes_rf2_100,'ER')
int3 = print_intersection(genes_da_100,genes_dfs3_100,genes_rf3_100,'HER2')

np.matrix([int1,int2,int3])


#############################################################################
'''Logistic Regression DSF'''

train = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\train_metabric.csv',index_col=False)
test = pd.read_csv('C:\\Users\\raque\\Documents\\SFU\\Machine Learning\\test_tcga.csv', index_col=False)

'''Removing rows with NaN clinical information'''
train= train.dropna()
test = test.dropna()

test = test.loc[(test.ER_status=='Positive')|(test.ER_status=='Negative')]
test = test.loc[(test.HER2_status=='Positive')|(test.HER2_status=='Negative')]

'''Split Dataset into clinical information and genes'''
train_c = train.loc[:,['Patient','overall_survival', 'oncotree_code', 'ER_status', 'HER2_status',
       'overall_survival_months','tumor_stage']]
test_c = test.loc[:,['Patient','overall_survival', 'oncotree_code', 'ER_status', 'HER2_status',
       'overall_survival_months','tumor_stage'] ]      

train_g = train.drop(['Patient','overall_survival', 'oncotree_code', 'ER_status', 'HER2_status',
       'overall_survival_months','tumor_stage'],axis=1)
test_g = test.drop(['Patient','overall_survival', 'oncotree_code', 'ER_status', 'HER2_status',
       'overall_survival_months','tumor_stage'],axis=1)


'''Change string to 0 or 1'''
train_c['overall_survival'] = train_c['overall_survival'].replace({'DECEASED':0,'LIVING':1})
train_c['ER_status'] = train_c['ER_status'].replace({'Negative':0,'Positive':1})
train_c['HER2_status'] = train_c['HER2_status'].replace({'Negative':0,'Positive':1})
test_c['overall_survival'] = test_c['overall_survival'].replace({'DECEASED':0,'LIVING':1})
test_c['ER_status'] = test_c['ER_status'].replace({'Negative':0,'Positive':1})
test_c['HER2_status'] = test_c['HER2_status'].replace({'Negative':0,'Positive':1})

train_c = train_c.as_matrix()
test_c = test_c.as_matrix()

'''
top100: list of genes and importance
train_ci and test_ci from as_matrix()
top100 = genes_da_100
train = train_g
test = test_g
train_ci = train_c[:,1]
test_ci = test_c[:,1]

'''

def logistic_regression(top100,train,test,train_ci,test_ci):
    genes = np.array(top100['genes'])
    train = train.loc[:,genes]
    test = test.loc[:,genes]
    train = train.dropna(axis=1)
    test = test.dropna(axis=1)
    train = train.as_matrix()
    test = test.as_matrix()
        
    train_ci = train_ci.astype('float')
    test_ci = test_ci.astype('float')
    #cross_validation_train = []
    #cross_validation_test = []
    accuracy_train = []
    accuracy_test = []
    
    '''Logisti Regression accucacy and cross-validation'''
    logreg = lm.LogisticRegression()
    logreg.fit(train, train_ci)
    #cross_validation_train.append(cv.cross_val_score(logreg,train_f , y_train).mean())
    accuracy_train.append(logreg.score(train, train_ci))
    #cross_validation_test.append(cv.cross_val_score(logreg,test_f,y_test).mean())
    accuracy_test.append(logreg.score(test, test_ci))
    
    print('accuracy in training and testing set')
    print(accuracy_train,'\n', accuracy_test)
    return [accuracy_train,accuracy_test]

columns = [1,3,4]
#Overall Survival
dfs1 = logistic_regression(genes_dfs1_100,train_g,test_g,train_c[:,1],test_c[:,1])
rf1 = logistic_regression(genes_rf1_100,train_g,test_g,train_c[:,1],test_c[:,1])
da1 = logistic_regression(genes_da_100,train_g,test_g,train_c[:,1],test_c[:,1])

#ER
dfs2 = logistic_regression(genes_dfs2_100,train_g,test_g,train_c[:,3],test_c[:,3])
rf2 = logistic_regression(genes_rf2_100,train_g,test_g,train_c[:,3],test_c[:,3])
da2 = logistic_regression(genes_da_100,train_g,test_g,train_c[:,3],test_c[:,3])

#HER
dfs3 = logistic_regression(genes_dfs2_100,train_g,test_g,train_c[:,4],test_c[:,4])
rf3 = logistic_regression(genes_rf2_100,train_g,test_g,train_c[:,4],test_c[:,4])
da3 = logistic_regression(genes_da_100,train_g,test_g,train_c[:,4],test_c[:,4])
 

genes_da_100['genes'] = genes_da_100['genes'].replace('03-Sep','Sep03')
train_g  = train_g.rename(columns={'03-Sep':'Sep03'})


#using 4 genes that were present in the intesections (1 for OR, 2 for ER and 1 to HER2)
genes_4 = genes_da_100[0:4]
genes_4.genes = ['CLDN11','PGR','TFF1','MUCL1']
logistic_regression(genes_4,train_g,test_g,train_c[:,1],test_c[:,1])
logistic_regression(genes_4,train_g,test_g,train_c[:,3],test_c[:,3])
logistic_regression(genes_4,train_g,test_g,train_c[:,4],test_c[:,4])






