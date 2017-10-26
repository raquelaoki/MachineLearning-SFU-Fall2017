#Class: Machine Learning - SFU - Fall 2017
#Assigment 2 
#Author: Raquel Aoki
#Date: 2017/10/24

import numpy 
import pandas
#import tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#https://svaante.github.io/decision-tree-id3/id3.html#id3.id3.Id3Estimator.fit

#reading dataset
pandas.set_option('display.max_columns', None)

#Question 4
print('\n QUESTION 4')

#1) Data Preprocessing
bd = pandas.DataFrame.from_csv('/home/raquel/Documentos/SFU/Machine Learning/preprocessed_datasets2.csv')
resp = bd['sum_7yr_GP']
bd = bd.drop('sum_7yr_GP',axis=1)
#intercept = pandas.Series(numpy.repeat(1, bd.shape[0]))
#resp = pandas.concat([resp,intercept],axis=1)
#resp = resp.rename(columns={0:'intecept'})
bd = pandas.concat([resp,bd],axis=1)
bd = bd[(bd.DraftYear == 2004) | (bd.DraftYear == 2005) |(bd.DraftYear == 2006)|(bd.DraftYear == 2007)]

#a) drop columns that won't be used
bd=bd.drop(['GP_greater_than_0','PlayerName', 'Country','sum_7yr_TOI','Overall'],axis=1)
print('a) Dropping columns GP_greater_than_0,PlayerName, Country,sum_7yr_TOI,Overall\n')


#b) dummy variables
colnames_cat = ['country_group', 'Position']
dummy = pandas.get_dummies(bd[colnames_cat])
DraftYear = bd['DraftYear']
bd=bd.drop(['country_group', 'Position','DraftYear'],axis=1)
bd = pandas.concat([DraftYear,bd, dummy], axis=1)
print('b) Dummy variables to country group and position\n')


#c) add quadratic interactions terms 
#There are 22 varibles+intercept, Thys, the new dataset will have 22+ 22*11/2 = 253 variables
col_size = bd.shape[1]
for i in numpy.arange(3,col_size):
    for j in numpy.arange((i+1),col_size):                
        aux_bd = bd[bd.columns[i]]*bd[bd.columns[j]]
        aux_bd = aux_bd.rename(str(bd.columns[i]) + '-' + str(bd.columns[j]))
        bd = pandas.concat([bd,aux_bd],axis = 1)
print('c) Add quadratic interactions terms. There are 22 variables -> 22 + 22*11/2 = 253 features + sum_7yr_GP + DraftYear')
print( 'Dataset shape',bd.shape,'\n')
#print( 'testing shape',test.shape)

#d) Standardize Predictors
#dummy variables are in columns[17:24]
col = list(range(2,17))+list(range(25,bd.shape[1]))
for i in col:
    bd[bd.columns[i]]= (bd[bd.columns[i]]-numpy.mean(bd[bd.columns[i]]))/(numpy.sqrt(numpy.var(bd[bd.columns[i]])))
print('d) Standardize Predictors - except dummy variables')
#dropping columns that are interaction between diferent levels of a same feature
#example: interaction between country_group_CAN and country_group_EURO 
bd = bd.dropna(axis=1, how='all')

##dropping draftyear
train = bd[(bd.DraftYear == 2004) | (bd.DraftYear == 2005) |(bd.DraftYear == 2006)]
test = bd[bd.DraftYear == 2007]
train = train.drop('DraftYear',axis=1)
test = test.drop('DraftYear',axis=1)
print('\n\n.... Split into training and testing set + dropping Draftyear\n\n')


#2)Evaluating a weight vector
print('2)Evaluating a weight vector: function evaluation(weight,dataset,lambda)\n')
def evaluation(weight,dataset,lambda1):
    xw =  numpy.zeros((1,1))
    w2 = 0
    for i in range(len(weight)):
        xw = numpy.array(xw)+ weight[i]*numpy.array(dataset[dataset.columns[i+1]])
        w2 = w2+weight[i]*weight[i]
    #sel = square-error loss
    sel = sum(numpy.square(numpy.array(dataset[dataset.columns[0]])-numpy.array(xw)))
    sel = (sel/2)+(lambda1*w2/2)
    #sel = sum((sel/2),(lambda1*w2/2))
    return sel

def test_evaluation(lambda1):
    weight = [2,3,-1]
    dataset = numpy.matrix([[6,-1,15,14],[5,10,14,11],[11,-2,-3,5]])
    index = [1,2,3]
    dataset= pandas.DataFrame(dataset, index=index)  
    print(evaluation(weight,dataset,lambda1))
    print('right value: 1743')

#test_evaluation(0.01)

#3) Finding a weight vector
print('3)Finding a weight vector: function finding_weight(dataset,lambda)\n')
def finding_weight(dataset,lambda1):
    y = dataset[dataset.columns[0]]
    x = dataset.drop(dataset.columns[0],axis=1)
    w = numpy.dot(numpy.dot(numpy.linalg.inv(lambda1*numpy.identity(x.shape[1])+numpy.dot(x.transpose(),x)),x.transpose()),y)
    w = numpy.squeeze(numpy.asarray(w))
    return w

def test_finding_weight(lambda1):
    w_true = numpy.matrix([[2],[3],[-1]])
    data = numpy.matrix([[12,-5,8,2],[6,8,-1,7],[-6,6,-1,15]])
    data = pandas.DataFrame(data,index=[0,1,2])
    w = finding_weight(data,lambda1)
    print('Answer:', w )
    print('True:',w_true)

#test_finding_weight(0.01)

#exponential grid search
lambda2 = [0.01,0.1,1,10,100,1000,10000,100000]
#lambda2 = [1000,100,10,1,0.1]
train = train.sample(frac=1).reset_index(drop=True) #train order is random
cross_val0 = [0,64,64,64,64,64,64,64,63,63,63] #10 folds (7 size 64 and 3 with size 63)
cross_val = numpy.cumsum(cross_val0)
cross_val_error = []
testing_error =[]

for j in range(len(lambda2)):
    cross_val_error_test = []
    w_est_test = []
    for i in range(10):
        cv_test  = train[cross_val[i]:cross_val[i+1]] #cross-validation test
        cv_train = train.drop(train.index[[numpy.arange(cross_val[i],cross_val[i+1])]]) #delete cross-validation test from training
        w_est = finding_weight(cv_train,lambda2[j]) #finding w in 9 fold-train
        cross_val_error_test.append(evaluation(w_est,cv_test,lambda2[j])) #error in 1 fold-test
        
    cross_val_error.append(numpy.mean(cross_val_error_test)) #average error in the 10 cross-validation
    w_est_test = finding_weight(train,lambda2[j]) #finding w in the entire training set
    testing_error.append(evaluation(w_est_test,test,lambda2[j])) #finding error in the entire testing set

    
#print(cross_val_error)
#print(testing_error)

#plot
#print(testing_error)
plt.semilogx(lambda2, cross_val_error,label='Squared-error loss - Cross Validation')
plt.semilogx(lambda2[5],testing_error[5],marker='o',color='r',label="Best Lambda")
plt.semilogx(lambda2, testing_error,label='squared-error loss - Testing set')
plt.legend()
plt.xlabel('Lambda')
plt.ylabel('Sum Squared Error')
plt.show()


#2) Results for lambda = 1000
w = finding_weight(train,lambda2[5])
train1 = train.copy()
train1 = train1.drop([train.columns[0]], axis=1)
features = train1.columns
w = numpy.absolute(w)

w1 = pandas.DataFrame(w,index=features)
w1 = w1.sort_values([0], axis=0,ascending=False)


#erro = evaluation(w,test,lambda2[5])

##test_evaluation(1)
#Question 2: Dataset
#test and training set

#train.to_csv('a2_train_weka.csv')
#test.to_csv('a2_test_weka.csv')

#train.to_csv('a2_train_reg.csv')
#test.to_csv('a2_test_reg.csv')

#exec(open("assignment2_raquelaoki.py").read())




