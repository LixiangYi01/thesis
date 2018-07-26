import csv
import numpy as np
from numpy.random import seed
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import regularizers
from keras.utils import to_categorical
from keras import initializers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow import set_random_seed

# Models

def baseline_model(l2=0.0,classes=20):
    input=layers.Input(shape=(90,))
    output=layers.Dense(classes,activation='softmax',
                          kernel_regularizer=regularizers.l2(l2),
                          bias_regularizer=regularizers.l2(l2))(input)
    model=models.Model(inputs=input,output=output)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def rnn_model(timesteps,classes,filters=16,kernel_size=3,pool_size=2,units=16,
              dropout=0.0,input_shape=20,l2=0.0):
    input=layers.Input(shape=(timesteps,input_shape))
    conv=layers.Conv1D(filters,kernel_size,padding='same',activation='relu')(input)
    conv=layers.MaxPooling1D(pool_size)(conv)
    conv=layers.Dropout(dropout)(conv)
    conv=layers.Conv1D(units,kernel_size,padding='same',activation='relu')(conv)
    gru=layers.Bidirectional(layers.GRU(units,dropout=dropout,
                           recurrent_dropout=dropout,
                           recurrent_regularizer=regularizers.l2(l2),
                           kernel_regularizer=regularizers.l2(l2),
                           bias_regularizer=regularizers.l2(l2),
                           return_sequences=True))(conv)
    attention=layers.Dense(2*units,activation='tanh',
                             kernel_regularizer=regularizers.l2(l2),
                             bias_regularizer=regularizers.l2(l2))(gru)
    attention=layers.Dropout(dropout)(attention)
    attention=layers.Dense(1,use_bias=False,
                             kernel_regularizer=regularizers.l2(l2),
                             bias_regularizer=regularizers.l2(l2))(attention)
    attention=layers.Flatten()(attention)
    attention=layers.Activation('softmax')(attention)
    attention=layers.Dropout(dropout)(attention)
    attention=layers.dot([gru,attention],axes=1)
    output=layers.Dense(classes,activation='softmax',
                          kernel_regularizer=regularizers.l2(l2),
                          bias_regularizer=regularizers.l2(l2))(attention)
    model=models.Model(inputs=input,output=output)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Loading

def load_data(file):
    loaded=np.load(file)
    X_train=loaded['X_train']
    X_test=loaded['X_test']
    y_train=loaded['y_train']
    y_test=loaded['y_test']
    return X_train,X_test,y_train,y_test

# Holdout

def holdout(id,build_model,preprocess,X,y,epochs=5000,batch_size=128,test_size=0.2):
    # Split off validation set
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=test_size,random_state=999,stratify=y)
    # Preprocessing
    preprocess(X_train,X_val)
    # One-hot encode targets
    y_train=to_categorical(y_train)
    y_val=to_categorical(y_val)
    # Fit model
    model=build_model()
    print(model.summary())
    history=model.fit(X_train,y_train,validation_data=(X_val,y_val),
                        epochs=epochs,batch_size=batch_size,verbose=1)
    # Evaluation
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    # Save all results
    np.savez_compressed(id+'_holdout_loss.npz',train=np.array(loss),val=np.array(val_loss))
    np.savez_compressed(id+'_holdout_acc.npz',train=np.array(acc),val=np.array(val_acc))
    # Return
    return acc,val_acc,loss,val_loss

def plot_loss(id,loss,val_loss,epochs):
    epochs=range(1,epochs+1)
    plt.clf()
    plt.plot(epochs,loss,'ro',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(id+'_holdout_loss.png')

def plot_acc(id,acc,val_acc,epochs):
    epochs=range(1,epochs+1)
    plt.clf()
    plt.plot(epochs,acc,'ro',label='Training Accuracy')
    plt.plot(epochs,val_acc,'b',label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(id+'_holdout_acc.png')

def run_holdout(data,build_model,preprocess,epochs=3000,row):
    X_train,X_test,y_train,y_test=load_data(data)
    acc,val_acc,loss,val_loss=holdout(row[0],build_model,preprocess,X_train,y_train,epochs=epochs)
    plot_loss(row[0],loss,val_loss,epochs)
    plot_acc(row[0],acc,val_acc,epochs)
    min_val_loss=np.min(val_loss)
    max_val_acc=np.max(val_acc)
    loss_epoch=np.argmin(val_loss)+1
    acc_epoch=np.argmax(val_acc)+1
    # Write results
    row+=['%0.4f'%val_loss[-1],'%0.4f'%min_val_loss,loss_epoch]
    row+=['%0.4f'%val_acc[-1],'%0.4f'%max_val_acc,acc_epoch]
    with open('results.csv','a') as f_out:
        writer=csv.writer(f_out)
        writer.writerow(row)
        f_out.flush()
    return acc,val_acc,loss,val_loss

# K-Fold CV

def cv(build_model,preprocess,X,y,k=10,epochs=5000,batch_size=128):
    all_val_acc=[]
    all_val_loss=[]    
    all_last_val_acc=[]
    all_last_val_loss=[]
    all_epochs=[]
    kf=StratifiedKFold(n_splits=k)
    i=1
    for train_index,val_index in kf.split(X,y):
        print('\nFold #',i)
        # Preprocessing
        X_train_fold,X_val_fold=X[train_index],X[val_index]
        y_train_fold,y_val_fold=y[train_index],y[val_index]
        preprocess(X_train_fold,X_val_fold)
        # One-hot encode targets
        y_train_fold=to_categorical(y_train_fold,36)
        y_val_fold=to_categorical(y_val_fold,36)
        # Fit model
        model=build_model()
        history=model.fit(X_train_fold,y_train_fold,validation_data=(X_val_fold,y_val_fold),
                            epochs=epochs,batch_size=batch_size,verbose=0)
        # Evaluation
        acc=history.history['acc']
        val_acc=history.history['val_acc']        
        loss=history.history['loss']
        val_loss=history.history['val_loss']
        all_val_acc.append(val_acc)
        all_val_loss.append(val_loss)
        all_last_val_loss.append(val_loss[-1])
        all_last_val_acc.append(val_acc[-1])
        print('Training loss:',loss[-1],'. Validation loss:',val_loss[-1])
        print('Training accuracy:',acc[-1],'. Validation accuracy:',
              val_acc[-1])
        all_epochs.append(np.argmin(val_loss))
        i+=1
    all_epochs=np.array(all_epochs)
    mean_epochs,std_epochs=np.mean(all_epochs),np.std(all_epochs)
    all_last_val_acc=np.array(all_last_val_acc)
    mean_val_acc,std_val_acc=np.mean(all_last_val_acc),np.std(all_last_val_acc)
    all_last_val_loss=np.array(all_last_val_loss)
    mean_val_loss,std_val_loss=np.mean(all_last_val_loss),np.std(all_last_val_loss)
    return mean_val_loss,std_val_loss,mean_val_acc,std_val_acc,mean_epochs,std_epochs,all_val_loss,all_val_acc

def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous * factor+point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_loss_histories(id,all_val_loss,epochs=5000):
    avg_val_loss=[np.mean([x[i] for x in all_val_loss]) for i in range(epochs)]
    smooth_val_loss=smooth_curve(avg_val_loss)
    plt.clf()
    plt.plot(range(1,len(smooth_val_loss)+1),avg_val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.savefig(id+'_cv_loss.png')

def plot_acc_histories(id,all_val_acc,epochs=5000):
    avg_val_acc=[np.mean([x[i] for x in all_val_acc]) for i in range(epochs)]
    smooth_val_acc=smooth_curve(avg_val_acc)
    plt.clf()
    plt.plot(range(1,len(smooth_val_acc)+1),avg_val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig(id+'_cv_acc.png')

def run_cv(data,build_model,preprocess,epochs=5000,k=10,row=None):
    X_train,X_test,y_train,y_test=load_data(data)
    mean_val_loss,std_val_loss,mean_val_acc,std_val_acc,mean_epochs,std_epochs,all_val_loss,all_val_acc=cv(build_model,preprocess,X_train,y_train,k=k,epochs=epochs)
    plot_loss_histories(row[0],all_val_loss,epochs)
    plot_acc_histories(row[0],all_val_acc,epochs)
    # Write results
    row+=['%0.4f'%mean_val_loss,'%0.4f'%std_val_loss]
    row+=['%0.4f'%mean_val_acc,'%0.4f'%std_val_acc]
    row+=['%0.4f'%mean_epochs,'%0.4f'%std_epochs]
    with open('results_baseline.csv','a') as f_out:
        writer=csv.writer(f_out)
        writer.writerow(row)
        f_out.flush()

# Prediction

def run_predict(data,build_model,preprocess,epochs=3000,batch_size=128,row=None):
    X_train,X_test,y_train,y_test=load_data(data)
    tr_time=X_train.shape[1]
    tst_time=X_test.shape[1]
    if tr_time > tst_time:
        b=np.zeros((X_train.shape[0],tr_time - tst_time,X_train.shape[2]))
        X_test=np.append(X_test,b,axis=1)
    elif tst_time > tr_time:
        print(tst_time - tr_time)
        b=np.zeros((X_train.shape[0],tst_time - tr_time,X_train.shape[2]))
        X_train=np.append(X_train,b,axis=1)
    # Preprocessing
    preprocess(X_train,X_test)
    # One-hot encode targets
    y_train=to_categorical(y_train)
    # Fit model
    model=build_model()
    history=model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
    # Evaluation
    y_pred=model.predict(X_test)
    test_acc=np.sum(y_test==(y_pred.argmax(axis=1)))/len(y_test)
    print('Accuracy on test set:',test_acc)
    row+=[test_acc]
    with open('results_predict.csv','a') as f_out:
        writer=csv.writer(f_out)
        writer.writerow(row)
        f_out.flush()

# Preprocessing

def standardize_baseline(X_train,X_test,col):
    mean=X_train[:,col].mean()
    std=X_train[:,col].std()
    X_train[:,col]-=mean
    X_train[:,col]/=std
    X_test[:,col]-=mean
    X_test[:,col]/=std

def preprocess_baseline(X_train,X_test):
    for i in range(90):
        standardize_baseline(X_train,X_test,i)

def standardize_rnn(X_train,X_test,col):
    mean=X_train[:,:,col].mean()
    std=X_train[:,:,col].std()
    X_train[:,:,col]-=mean
    X_train[:,:,col]/=std
    X_test[:,:,col]-=mean
    X_test[:,:,col]/=std
    return X_train,X_test

def preprocess_rnn(X_train,X_test):
    # Standardize duration and transition
    X_train,X_test=standardize_rnn(X_train,X_test,0)
    X_train,X_test=standardize_rnn(X_train,X_test,1)
    return X_train,X_test

def run_rnn_holdout(id,filters=32,units=32,dropout=0.5,l2=0,epochs=3000,seed=999,
              timesteps=1000,f=None,classes=36,kernel_size=3,pool_size=2):
    def build_model():
        return rnn_model(timesteps=timesteps,classes=classes,input_shape=num_features,
                         units=units,filters=filters,dropout=dropout,l2=l2,
                         kernel_size=kernel_size,pool_size=pool_size)
    # Set seed
    seed(16)
    set_random_seed(16)
    # Which input file to use.
    if f==None:
      f='rnn_villani_mixed.npz'
    # Row prefix
    row=[id,'ConvRNN',num_features,units,filters,dropout,l2,epochs]
    # Perform holdout
    run_holdout(f,build_model,preprocess_rnn,epochs=epochs,row=row)

def run_rnn_predict(id,filters=64,units=64,dropout=0.5,l2=0.01,seed=999,
        f=None,epochs=3000,timesteps=1000,classes=36,kernel_size=3,pool_size=2):
    def build_model():
        return rnn_model(timesteps=timesteps,classes=classes,
                input_shape=num_features,
                units=units,filters=filters,dropout=dropout,l2=l2,
                kernel_size=kernel_size,pool_size=pool_size)
    # Set seed
    seed(16)
    set_random_seed(16)
    # Which input file to use.
    if f==None:
        f='rnn_villani_mixed.npz'
    # Run
    row=[id,f,'ConvRNN',num_features,units,filters,dropout,l2,epochs]
    run_predict(f,build_model,preprocess_rnn,epochs=epochs,row=row)

def run_baseline_cv(id,l2=0.0,epochs=5000,k=10,classes=36,f=None,seed=999):
    def build_model():
        return baseline_model(l2=l2,classes=classes)
    # Set seed
    seed(16)
    set_random_seed(16)
    # Which input file to use.
    if f==None:
        f='baseline_villani_mixed.npz'
    # Row prefix
    row=[id,'Baseline',f,l2,epochs,k]
    # Perform cross-validation
    run_cv(f,build_model,preprocess_baseline,epochs=epochs,k=k,row=row)

def run_baseline_predict(id,l2=0,epochs=3000,f=None,classes=36,seed=999):
    def build_model():
        return baseline_model(l2=l2,classes=classes)
    # Set seed
    seed(16)
    set_random_seed(16)
    # Which input file to use
    if f==None:
        f='baseline_villani_mixed.npz'
    # Run
    row=[id,f,'Baseline',num_features,units,filters,dropout,l2,epochs]
    run_predict(f,build_model,preprocess_baseline,epochs=epochs,row=row)
