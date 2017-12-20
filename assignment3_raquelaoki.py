#Class: Machine Learning - SFU - Fall 2017
#Assigment 3 
#Author: Raquel Aoki
#Date: 2017/11/17

import numpy 
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.special as sps
import matplotlib.pyplot as plt


#reading dataset
pandas.set_option('display.max_columns', None)

#Data Preprocessing
print('\n Data Preprocessing')

bd = pandas.DataFrame.from_csv('/home/raquel/Documentos/SFU/Machine Learning/preprocessed_datasets2.csv')
#bd = pandas.DataFrame.from_csv('C:\\Users\\raoki\\Downloads\\preprocessed_datasets2.csv')
resp = bd['sum_7yr_GP']
bd = bd.drop('sum_7yr_GP',axis=1)
bd = pandas.concat([resp,bd],axis=1)
bd = bd[(bd.DraftYear == 2004) | (bd.DraftYear == 2005) |(bd.DraftYear == 2006)|(bd.DraftYear == 2007)]

#drop columns that won't be used
bd=bd.drop(['PlayerName', 'Country','sum_7yr_TOI','Overall'],axis=1)
print('a) Dropping columns PlayerName, Country,sum_7yr_TOI,Overall\n')

#dummy variables
colnames_cat = ['country_group', 'Position']
dummy = pandas.get_dummies(bd[colnames_cat])
DraftYear = bd['DraftYear']
bd=bd.drop(['country_group', 'Position','DraftYear'],axis=1)
bd = pandas.concat([DraftYear,bd, dummy], axis=1)
print('b) Dummy variables to country group and position\n')


#standart 
col=['DraftAge', 'Height',  'Weight',  'CSS_rank' ,'rs_GP',  'rs_G',  'rs_A' , 'rs_P',  'rs_PIM',  'rs_PlusMinus', 
     'po_GP',  'po_G',  'po_A',  'po_P',  'po_PIM' , 'country_group_CAN',  'country_group_EURO', 
     'country_group_USA',  'Position_C' , 'Position_D' , 'Position_L',  'Position_R' ]
for i in col:
    bd[i]= (bd[i]-numpy.mean(bd[i]))/(numpy.sqrt(numpy.var(bd[i])))

#Question 2
print('Question 2\n')

print('\n\n.... Split into training and testing set + dropping Draftyear\n\n')

##dropping draftyear
train = bd[(bd.DraftYear == 2004) | (bd.DraftYear == 2005) |(bd.DraftYear == 2006)]
test = bd[bd.DraftYear == 2007]
train = train.drop(['DraftYear','sum_7yr_GP'],axis=1)
test = test.drop(['DraftYear','sum_7yr_GP'],axis=1)

GP_greater_than_0 = train['GP_greater_than_0']
#train=train.drop(['GP_greater_than_0','po_GP',  'po_G',  'po_A',  'po_P',  'po_PIM'],axis=1)
train=train.drop(['GP_greater_than_0'],axis=1)
col1 = pandas.Series(numpy.zeros((train.shape[0]))+1, index = train.index)
train = pandas.concat([col1,train,GP_greater_than_0], axis=1)

GP_greater_than_0 = test['GP_greater_than_0']
#test=test.drop(['GP_greater_than_0','po_GP',  'po_G',  'po_A',  'po_P',  'po_PIM'],axis=1)
test=test.drop(['GP_greater_than_0'],axis=1)
col1 = pandas.Series(numpy.zeros((test.shape[0]))+1, index = test.index)
test = pandas.concat([col1,test,GP_greater_than_0], axis=1)

train['GP_greater_than_0'] = train['GP_greater_than_0'].replace('no',0)
train['GP_greater_than_0'] = train['GP_greater_than_0'].replace('yes',1)
test['GP_greater_than_0'] = test['GP_greater_than_0'].replace('no',0)
test['GP_greater_than_0'] = test['GP_greater_than_0'].replace('yes',1)




# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
numpy.random.seed(42)
train = numpy.random.permutation(train)
test = numpy.random.permutation(test)

# Data matrix, with column of ones at end.
X = train[:,0:23] #23
Xtest = test[:,0:23] #23
# Target values, 0 for class 1, 1 for class 2.
t = train[:,23]
ttest = test[:,23]
# For plotting data
class1 = numpy.where(t==0)
X1 = X[class1]
class2 = numpy.where(t==1)
X2 = X[class2]

n_train = t.size

# Error values over all iterations.
all_errors = dict()
accuracy = []

for eta in etas:
    # Initialize w.
    w = numpy.zeros((23))
    w[0] = w[0]+0.1
    e_all = []
    for iter in range (0, max_iter):
        for n in range (0, n_train):
            # Compute output using current w on sample x_n.
            y = sps.expit(numpy.dot(X[n,:],w))
        
            # Gradient of the error, using Assignment result
            grad_e = (y - t[n])*X[n,:]
        
            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w = w - eta*grad_e
              
    
        # Compute error over all examples, add this error to the end of error vector.
        # Compute output using current w on all data X.
        y = sps.expit(numpy.dot(X,w))
      
        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -numpy.mean(numpy.multiply(t,numpy.log(y)) + numpy.multiply((1-t),numpy.log(1-y)))        
        #e = -(sum(t[t==1]*numpy.log(y[t==1]))+ sum(numpy.log(1-y[t==0])*(1-t[t==0])))/len(t)
        e_all.append(e)
    
        # Print some information.
        #print('eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}, w={3}',format(eta, iter, e, w.T))
      
        # Stop iterating if error doesn't change more than tol.
        if iter>0:
            if numpy.absolute(e-e_all[iter-1]) < tol:
                break
                
        all_errors[eta] = e_all
    
    ytest = sps.expit(numpy.dot(Xtest[:],w))
    ytest[ytest>0.5] = 1
    ytest[ytest<=0.5] = 0
    accuracy.append(sum(1 for x,y in zip(ytest,ttest) if x == y) / len(ytest))

print(accuracy)
print(all_errors)
# Plot error over iterations for all etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
for eta in sorted(all_errors):
    plt.plot(all_errors[eta], label='sgd eta={}'.format(eta))
    
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression with SGD')
plt.xlabel('Epoch')
plt.axis([0, 25, 0,2])
plt.legend()
plt.show()

# Plot accuracy over etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
plt.plot(etas,accuracy)
plt.ylabel('Accuracy')
plt.title('Training logistic regression with SGD')
plt.xlabel('Etas')
plt.axis([0, 0.55, 0,1])
plt.legend()
plt.show()


#Dataset non standartized - result not good
#GD
# Error values over all iterations.
all_errors_gd = dict()
accuracy_gd = []

for eta in etas:
    # Initialize w.
    w = numpy.zeros((23))
    w[0] = w[0]+0.1
    e_all = []
    for iter in range (0, max_iter):
        # Compute output using current w on sample x_n.
        y = sps.expit(numpy.dot(X[:],w))
        
        # Gradient of the error, using Assignment result
        grad_e = numpy.mean(y - t[n])*X[n,:]
        
        # Update w, *subtracting* a step in the error derivative since we're minimizing
        w = w - eta*grad_e
              
        # Compute error over all examples, add this error to the end of error vector.
        # Compute output using current w on all data X.
        y = sps.expit(numpy.dot(X,w))
      
        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -numpy.mean(numpy.multiply(t,numpy.log(y)) + numpy.multiply((1-t),numpy.log(1-y)))        
        #e = -(sum(t[t==1]*numpy.log(y[t==1]))+ sum(numpy.log(1-y[t==0])*(1-t[t==0])))/len(t)
        e_all.append(e)
    
        # Print some information.
        #print('eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}, w={3}',format(eta, iter, e, w.T))
      
        # Stop iterating if error doesn't change more than tol.
        if iter>0:
            if numpy.absolute(e-e_all[iter-1]) < tol:
                break
                
        all_errors_gd[eta] = e_all
    
    ytest = sps.expit(numpy.dot(Xtest[:],w))
    ytest[ytest>0.5] = 1
    ytest[ytest<=0.5] = 0
    accuracy_gd.append(sum(1 for x,y in zip(ytest,ttest) if x == y) / len(ytest))

print(accuracy_gd)
print(all_errors_gd)

#Question 5
print('\n QUESTION 5')

##dropping draftyear
train = bd[(bd.DraftYear == 2004) | (bd.DraftYear == 2005) |(bd.DraftYear == 2006)]
test = bd[bd.DraftYear == 2007]
train = train.drop(['DraftYear','GP_greater_than_0'],axis=1)
test = test.drop(['DraftYear','GP_greater_than_0'],axis=1)
print('\n\n.... Split into training and testing set + dropping Draftyear\n\n')

# split into input (X) and output (Y) variables

X_train = train.loc[:,train.columns[1:25]]
Y_train = train.loc[:,train.columns[0]]
X_test = test.loc[:,test.columns[1:25]]
Y_test = test.loc[:,test.columns[0]]

X_train = X_train.as_matrix()
Y_train = Y_train.as_matrix()
X_test = X_test.as_matrix()
Y_test = Y_test.as_matrix()

#Models 
#Test
#Testing using linear and relu
#Testing with 0, 1 or 2 hidden layer 
#Change the number of units on hidden layer

#Optional
#sigmoid on hidden layer (not output)
#learning rates or ADAM
#batch normalization and Dropout

parameters = [
            (i,[j1,j2,j3],[k1,k2,k3])
                for i in [0,1,2]
                for j3 in [6,12,24,48]
                for j2 in [6,12,24]
                for j1 in [6,12]
                for k3 in ['sigmoid','relu','linear']
                for k2 in ['sigmoid','relu','linear']                
                for k1 in ['relu','linear']
        ]
print(parameters[0:3])


#Functions with 0,1 or 2 layers 
#12,24,48 units
#relu or linear
def model(layer,units,act):
    # create model
    model = Sequential()
    if layer==0:
        model.add(Dense(1,input_dim=22))
    elif layer == 1: 
        model.add(Dense(units[0],input_dim=22))        
        model.add(Activation(act[1]))
        model.add(Dense(units=1))
    elif layer == 2:
        model.add(Dense(units[1],input_dim=22))        
        model.add(Activation(act[1]))
        model.add(Dense(units=units[0]))
        model.add(Activation(act[2]))
        model.add(Dense(units=1))
    else:
        model.add(Dense(units[2],input_dim=22))        
        model.add(Activation(act[0]))
        model.add(Dense(units=units[1]))
        model.add(Activation(act[1]))
        model.add(Dense(units=units[0]))
        model.add(Activation(act[2]))
        model.add(Dense(units=1))
        
    model.add(Activation(act[0]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

'''
error = []
for ind in range(len(parameters)):
    regr = model(parameters[ind][0],parameters[ind][1],parameters[ind][2])
    regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
    loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
    error.append(loss_and_metrics)
    if ind % 100 == 0:
        parameters2 = pandas.DataFrame(parameters[0:(ind+1)],error)
        parameters2.to_csv('a3_tune2.csv')
'''
        

'''model with dropout'''
def model_drop(layer,units,act):
    model = Sequential()
    model.add(Dense(units[1],input_dim=22))        
    model.add(Activation(act[2]))
    model.add(Dropout(0.1))
    model.add(Dense(units=units[0]))
    model.add(Activation(act[1]))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.add(Activation(act[0]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

error_drop = []
#1
regr = model_drop(2,[12,6],['linear', 'relu', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error_drop.append(loss_and_metrics)

regr = model_drop(2,[6,6],['relu', 'linear', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error_drop.append(loss_and_metrics)

regr = model_drop(2,[12,24],['linear', 'relu', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error_drop.append(loss_and_metrics)

regr = model_drop(2,[12,12],['linear', 'linear', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error_drop.append(loss_and_metrics)


regr = model_drop(2,[12,6],['linear', 'linear', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error_drop.append(loss_and_metrics)
print("Top models testing error - dropout",error_drop)

error = []
#1
regr = model(2,[12,6],['linear', 'relu', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error.append(loss_and_metrics)

regr = model(2,[6,6],['relu', 'linear', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error.append(loss_and_metrics)

regr = model(2,[12,24],['linear', 'relu', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error.append(loss_and_metrics)

regr = model(2,[12,12],['linear', 'linear', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error.append(loss_and_metrics)


regr = model(2,[12,6],['linear', 'linear', 'relu'])
regr.fit(X_train, Y_train, nb_epoch=200, validation_split=0.15, verbose = 50) #batch_size=5,
loss_and_metrics = regr.evaluate(X_test, Y_test, batch_size=256) #batch_size bigger faster
error.append(loss_and_metrics)
print("Top models testing error",error)

#exec(open("assignment3_raquelaoki.py").read())
