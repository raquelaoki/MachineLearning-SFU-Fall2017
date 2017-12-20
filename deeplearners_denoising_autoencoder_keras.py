# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:06:43 2017
Machine Learning Project: Feature Extraction in Genomic Datasets
@author: raquel
exec(open("ml_project_da.py").read())
"""

import pandas as pd
import numpy as np
from keras import layers
from keras.layers.core import Dense, Activation
from keras.models import Sequential
import time
import sklearn.linear_model as lm
import sklearn.cross_validation as cv
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


np.set_printoptions(threshold=200)


'''Reading Datasets'''
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

genes = train_g.columns
train_g = train_g.as_matrix()
test_g = test_g.as_matrix()

'''Add those parameters'''
learning_rate = 0.01 #not used
batch_size = 10
training_epochs = 500


'''Autoencoder'''
model = Sequential()
#Noise
model.add(layers.GaussianDropout(0.01,input_shape=(2520,)))
#encode 
model.add(Dense(100)) #,input_shape=(2520,)
model.add(Activation('sigmoid'))
#decode 
model.add(Dense(2520))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error',optimizer='adam')

'''Same parameters used in paper'''
start_time = time.time()
model.fit(train_g,train_g, nb_epoch=training_epochs,shuffle=True,batch_size=batch_size,validation_split=0.15)
elapsed_time = time.time() - start_time
loss_and_metrics = model.evaluate(test_g,test_g) #batch_size bigger faster

'''Short Representation - Features'''
#https://github.com/fchollet/keras/issues/41
model2 = Sequential()
model2.add(Dense(100 ,input_shape=(2520,), weights=model.layers[1].get_weights()))
model2.add(Activation('sigmoid'))
train_f = model2.predict(train_g)
test_f = model2.predict(test_g)


'''
The weights of each node have normal distribution around 0
w>2sigma or w<2sigma are genes import for a node
select genes with highest nodes importance 
'''
weight = np.asmatrix( model.layers[1].get_weights()[0])
importance = np.zeros(weight.shape)
for i in range(weight.shape[1]):
    mean = weight[:,i].mean()
    sd = np.std(weight[:,i])
    for j in range(weight.shape[0]):
        if (weight[j,i]>(mean+2*sd)) | (weight[j,i]<(mean-2*sd)):
            importance[j,i] = 1

sum_importance = np.sum(importance,axis=1)
data = {'genes':genes,  'importance':sum_importance}
feature_importance = pd.DataFrame(data)
feature_importance = feature_importance.sort_values('importance',ascending=False)
#feature_importance.to_csv('features_importance_DA_all.csv') 
feature_importance = feature_importance[0:100]
#feature_importance.to_csv('features_importance_DA.csv') 

'''Logistic Regression between lower dimension and clinical information'''
#not use oncotree_code
#later turmor_stage
train_c.head()
print(train_c['overall_survival'].value_counts())
print(train_c['ER_status'].value_counts())
print(train_c['HER2_status'].value_counts())

'''Change string to 0 or 1'''
train_c['overall_survival'] = train_c['overall_survival'].replace({'DECEASED':0,'LIVING':1})
train_c['ER_status'] = train_c['ER_status'].replace({'Negative':0,'Positive':1})
train_c['HER2_status'] = train_c['HER2_status'].replace({'Negative':0,'Positive':1})
test_c['overall_survival'] = test_c['overall_survival'].replace({'DECEASED':0,'LIVING':1})
test_c['ER_status'] = test_c['ER_status'].replace({'Negative':0,'Positive':1})
test_c['HER2_status'] = test_c['HER2_status'].replace({'Negative':0,'Positive':1})


order = train_c.columns
train_c = train_c.as_matrix()
test_c = test_c.as_matrix()

#categorical columns to test logistic regression
columns = [1,3,4]
cross_validation_train = []
cross_validation_test = []
accuracy_train = []
accuracy_test = []

'''Logisti Regression accucacy and cross-validation'''
for i in columns:
    y_train = train_c[:,i].astype('int')
    y_test  = test_c[:,i].astype('int')
    logreg = lm.LogisticRegression()
    logreg.fit(train_f, y_train)
    cross_validation_train.append(cv.cross_val_score(logreg,train_f , y_train).mean())
    accuracy_train.append(logreg.score(train_f , y_train))
    cross_validation_test.append(cv.cross_val_score(logreg,test_f,y_test).mean())
    accuracy_test.append(logreg.score(test_f,y_test))
    
print('cross_validation in training and testing set')
print(cross_validation_train,'\n',cross_validation_test)
print('accuracy in training and testing set')
print(accuracy_train,'\n', accuracy_test)


'''Linear Regression'''
y_train = train_c[:,5].astype('float')
y_test  = test_c[:,5].astype('float')
logreg = lm.LinearRegression()
logreg.fit(train_f, y_train)
y_train_pred = logreg.predict(train_f)
y_test_pred = logreg.predict(test_f)
print('Mean Squared error')
print(metrics.mean_squared_error(y_train, y_train_pred))
print(metrics.mean_squared_error(y_test, y_test_pred))
    
#graphics
plt.figure(10)
plt.rcParams.update({'font.size':15})
plt.plot(y_test_pred,y_test,'bs')
plt.title('TCGA')
plt.ylabel('Overall Survival - Months')
plt.xlabel('Overall Survival Predicted- Months')
plt.legend()
plt.show()

plt.figure(10)
plt.rcParams.update({'font.size':15})
plt.plot(y_train_pred,y_train,'bs')
plt.title('METABRIC')
plt.ylabel('Overall Survival - Months')
plt.xlabel('Overall Survival Predicted- Months')
plt.legend()
plt.show()

y_train.mean()
y_test.mean()





