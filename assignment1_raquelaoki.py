#Class: Machine Learning - SFU - Fall 2017
#Assigment 1 
#Author: Raquel Aoki
#Date: 2017/10/10

import numpy 
import pandas
from scipy.stats import norm

#reading dataset
pandas.set_option('display.max_columns', None)
bd = pandas.DataFrame.from_csv('/home/raquel/Documentos/SFU/Machine Learning/preprocessed_datasets.csv')

#Question 7
#function to calculate mean and variance(bias)
def mean_var(col):
    n = len(col)
    mean = sum(col)/n
    var =  sum((col-mean)*(col-mean))/n
    return (mean,var)

#results 
print('\nClass: CMPT725 - Machine Learning \nStudent: Raquel Aoki')
print('\n QUESTION 7')
print('Item 1 - Complete: Apply your program to the column Weight in the assignment \n dataset (see course website) and show the results.')
col1 = bd.Weight
item1 = mean_var(col1)
print("Meam Complete: " + str(item1[0]))
print('Variance Complete: '+str(item1[1]))

print('\n')
print('Item 2 - GP>0: Apply your program to the column Weight conditional on GP > 0 \n being true and show the results')
col2 = bd[bd.GP_greater_than_0=='yes'].Weight
item2 = mean_var(col2)
print("Meam GP>0: " + str(item2[0]))
print('Variance GP>0: '+str(item2[1]))

print('\n')
print('Item 3 - GP<= 0: Apply your program to the column Weight conditional on GP > 0 \n being false (i.e. GP = 0 ) and show the results.')
col3 = bd[bd.GP_greater_than_0=='no'].Weight
item3 = mean_var(col3)
print("Meam GP<=0: " + str(item3[0]))
print('Variance GP<=0: '+str(item3[1]))


#Question 8
print('\n QUESTION 8')
#drop columns that won't be used
bd=bd.drop(['sum_7yr_GP','sum_7yr_TOI','Country','po_PlusMinus'],axis=1)

#transforming categorical variables into dummy variables
colnames_cat = ['country_group', 'Position']
dummy = pandas.get_dummies(bd[colnames_cat])
bd=bd.drop(['country_group', 'Position','PlayerName'],axis=1)
bd = pandas.concat([bd, dummy], axis=1)

#creating the train and test datasets
train = bd[(bd.DraftYear == 2004) | (bd.DraftYear == 2005) |(bd.DraftYear == 2006)]
test = bd[bd.DraftYear == 2007]
test = test .sort_values('Overall') 



#spliting again the dataset to help at the implementation
trainY = train[train.GP_greater_than_0 == 'yes']
trainN = train[train.GP_greater_than_0 == 'no']
train_obs = train['GP_greater_than_0']
train = train.drop(['DraftYear','GP_greater_than_0'],axis=1)
trainY = trainY.drop(['DraftYear','GP_greater_than_0'],axis=1)
trainN = trainN.drop(['DraftYear','GP_greater_than_0'],axis=1)
test_obs = test['GP_greater_than_0']
test = test.drop(['DraftYear','GP_greater_than_0'],axis=1)

#columns with continuous and discrete variables
cont = list(range(0,15))
disc = list(range(16,trainY.shape[1]))

#accuracy function using equation 3
def accuracy(trainY_mean_cont, trainN_mean_cont,
             trainY_var_cont, trainN_var_cont,trainY_disc,trainN_disc,base, obs):                
                             
                 #dataset with the new values
                 base_y = base.as_matrix()
                 base_n = base.as_matrix()
                 ratio = base.as_matrix()/2                                
                 check = base.as_matrix()/2                 
                 
                 #ratio for continuous variables
                 for i in range(0, len(trainY_mean_cont)):
                    y = (base_y[:,i]-trainY_mean_cont[i])/numpy.sqrt(numpy.asarray(trainY_var_cont[i]))
                    n = (base_n[:,i]-trainN_mean_cont[i])/numpy.sqrt(numpy.asarray(trainN_var_cont[i]))
                    ratio[:,i] = numpy.log(norm.cdf([y.astype(float)])) - numpy.log(norm.cdf([n.astype(float)]))    
                    check[:,i] = norm.cdf([y.astype(float)])

                 #ratio for discrete variables
                 for i in range(len(trainY_mean_cont),len(trainY_mean_cont)+len(trainY_disc)):
                    y =  1 - base_y[:,i] - trainY_disc[i-16]
                    n =  1 - base_n[:,i] - trainN_disc[i-16]
                    y[y<0] = y[y<0]*(-1)
                    n[n<0] = n[n<0]*(-1)
                    ratio[:,i] =  numpy.log(y) - numpy.log(n)

                 #using log
                 log_ratio = ratio
                 soma_ratio = numpy.sum(log_ratio, axis = 1)+numpy.log(base_y.shape[1]) - numpy.log(base_n.shape[1])
                 predito = soma_ratio.copy()
                 predito[soma_ratio>0] = 1
                 predito[soma_ratio<=0] = 0
                 obs = obs.replace(['no','yes'],[0,1])
                 ac = pandas.crosstab(obs,predito, rownames=['true'], colnames=['predict'])
                 print(ac)
                 print(round(ac*100/len(obs),2),'\n')


def accuracy2(trainY_mean_cont, trainN_mean_cont,
             trainY_var_cont, trainN_var_cont,trainY_disc,trainN_disc,base, obs):                
                             
                 #dataset with the new values
                 base_y = base.as_matrix()
                 base_n = base.as_matrix()
                 ratio = base.as_matrix()/2                                
                 
                 #ratio for continuous variables
                 for i in range(0, len(trainY_mean_cont)):
                    y = (base_y[:,i]-trainY_mean_cont[i])/numpy.sqrt(numpy.asarray(trainY_var_cont[i]))
                    n = (base_n[:,i]-trainN_mean_cont[i])/numpy.sqrt(numpy.asarray(trainN_var_cont[i]))
                    ratio[:,i] = norm.cdf([y.astype(float)])/norm.cdf([n.astype(float)])

                 #ratio for discrete variables
                 for i in range(len(trainY_mean_cont),len(trainY_mean_cont)+len(trainY_disc)):
                    y =  1 - base_y[:,i] - trainY_disc[i-16]
                    n =  1 - base_n[:,i] - trainN_disc[i-16]
                    y[y<0] = y[y<0]*(-1)
                    n[n<0] = n[n<0]*(-1)
                    ratio[:,i] =  y/n

                 #using ratio
                 prod_ratio = numpy.prod(ratio, axis = 1)*base_y.shape[1]/base_n.shape[1]
                 predito = prod_ratio.copy()
                 predito[prod_ratio>0.5] = 1
                 predito[prod_ratio<=0.5] = 0
                 obs = obs.replace(['no','yes'],[0,1])
                 ac = pandas.crosstab(obs,predito, rownames=['true'], colnames=['predict'])
                 print(ac)
                 print(round(ac*100/len(obs),2),'\n')


                 
#FIRST MODEL
#creating arrays to keep the features of the districutions 
trainY_mean_cont = []
trainN_mean_cont = []
trainY_var_cont = []
trainN_var_cont = []
trainY_disc = []
trainN_disc = []
#for each continuos variable calculate the mean and var using the 
#function created at question 7
for i in range(0, trainY.shape[1]-dummy.shape[1]):
    resY = mean_var(trainY[trainY.columns[i]])
    trainY_mean_cont.append(resY[0])
    trainY_var_cont.append(resY[1])
    resN = mean_var(trainN[trainN.columns[i]])
    trainN_mean_cont.append(resN[0])
    trainN_var_cont.append(resN[1])   

#for each discrete variable calculate the probability of P(X1=1)
for i in range(trainY.shape[1]-dummy.shape[1],trainY.shape[1]):
    train[train.columns[i]]
    trainY_disc.append(trainY[trainY.columns[i]].sum()/trainY.shape[0]  )
    trainN_disc.append(trainN[trainN.columns[i]].sum()/trainN.shape[0] ) 

print('\nFirst Model - Bias variance \n')
print('Equation 2 \n')
accuracy(trainY_mean_cont, trainN_mean_cont,
             trainY_var_cont, trainN_var_cont,trainY_disc,trainN_disc,test, test_obs)    
print('\nEquation 3 \n')
accuracy2(trainY_mean_cont, trainN_mean_cont,
             trainY_var_cont, trainN_var_cont,
             trainY_disc,trainN_disc,test, test_obs)  

#corrigindo a variancia
def mean_var_notbias(col):
    n = len(col)
    mean = sum(col)/n
    var =  sum((col-mean)*(col-mean))/(n-1)
    return (mean,var)

#creating arrays to keep the features of the districutions 
trainY_var_cont_notbias = []
trainN_var_cont_notbias = []

#for each continuos variable calculate the mean and var using the 
#function created at question 7
for i in range(0, trainY.shape[1]-dummy.shape[1]):
    resY = mean_var_notbias(trainY[trainY.columns[i]])
    trainY_var_cont_notbias.append(resY[1])
    resN = mean_var_notbias(trainN[trainN.columns[i]])
    trainN_var_cont_notbias.append(resN[1])   

print('\nSecond Model - Bias variance \n')
print('Equation 2 \n')
accuracy(trainY_mean_cont, trainN_mean_cont,
             trainY_var_cont_notbias, trainN_var_cont_notbias,
             trainY_disc,trainN_disc,test, test_obs)    
print('\nEquation 3 \n')
accuracy2(trainY_mean_cont, trainN_mean_cont,
             trainY_var_cont_notbias, trainN_var_cont_notbias,
             trainY_disc,trainN_disc,test, test_obs)              

#
#exec(open("assignment1_raquelaoki.py").read())




