import codecs
import utility_functions_v2 as util
import load_buffalo_data as data_loader
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout,LSTM,TimeDistributed,GRU , Bidirectional,BatchNormalization,Activation
from tensorflow.keras.models import  Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from collections import Counter
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

participants = ["005","080","083","017","019"]

def auroc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.double)

def custom_loss(y_true,y_pred,clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)

def get_call_backs():
    early_stopping = EarlyStopping(monitor = 'val_loss',patience = 30, mode = "min")
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=20)
    return (early_stopping, learning_rate_reduction)

def build_cnn_lstm_model(rnn_type,rnn_state,num_conv_filters,num_rnn_cells,drop_out,input_shape):
    print("cnn_lstm configuration 1")
      
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_conv_filters, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))

    model.add(Flatten())

    model.add(BatchNormalization())
        
    model.add(Dense(num_rnn_cells,activation='relu')) 
    model.add(Dropout(0.3))
    model.add(Dense(num_rnn_cells,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_cnn_lstm_model_ver2(rnn_type,rnn_state,num_conv_filters,num_rnn_cells,drop_out,input_shape):
    print("cnn_lstm configuration 2")
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_conv_filters, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,dropout = drop_out)) 
      
    model.add(BatchNormalization())
        
    model.add(Dense(num_rnn_cells,activation='relu')) 
    model.add(Dropout(0.3))
    model.add(Dense(num_rnn_cells,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_cnn_lstm_model_ver3(rnn_type,rnn_state,num_conv_filters,num_rnn_cells,drop_out,input_shape):
    print("cnn_lstm configuration 3- CNN units: {} , LSTM units: {} ".format(num_conv_filters,num_rnn_cells))
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_conv_filters, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,dropout = drop_out))
          
    model.add(Flatten())

    model.add(BatchNormalization())
        
    model.add(Dense(num_rnn_cells,activation='relu')) 
    model.add(Dropout(0.3))
    model.add(Dense(num_rnn_cells,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_cnn_lstm_model_multiclass(rnn_type,rnn_state,num_conv_filters,num_rnn_cells,drop_out,input_shape,num_output):
    print("CNN LSTM for multiclass- classification- CNN units: {} , LSTM units: {} ".format(num_conv_filters,num_rnn_cells))
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_conv_filters, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,return_sequences=True,dropout = drop_out))
    model.add(rnn_type(num_rnn_cells,stateful = rnn_state,dropout = drop_out))
          
    model.add(Flatten())

    model.add(BatchNormalization())
        
    model.add(Dense(num_rnn_cells,activation='relu')) 
    model.add(Dropout(0.3))
    model.add(Dense(num_rnn_cells,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_output, activation='softmax'))
    return model


def train_model(model, optimizer, loss_func, epochs, batch_size, data, label, call_back, metrics = ['acc',auroc]):
    for i in range(10):
        data,label = shuffle(data, label)
    model.compile(optimizer = optimizer, 
                  loss = loss_func,
                  metrics = metrics)
    history = model.fit(data, label, epochs=epochs, 
                        callbacks= call_back, 
                        batch_size=batch_size, validation_split=0.3, shuffle=True)
    if metrics!=['acc',auroc]:
        plot_roc = False
    else:
        plot_roc = True
    plot_results(history, None, plot_roc).show()
    
    return history


def calculate_keystroke_scores(y_true,y_pred_class):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    FAR,FRR = util.calculate_FPR_FRR(tn, fp, fn, tp)
    EER = util.calculate_eer(y_true,y_pred_class)
    print("TN:    {},    FP:   {},   FN:   {},   TP:   {}".format(tn, fp, fn, tp))
    print("FAR:   {},   FRR:   {},   EER:   {}".format(FAR, FRR, EER))
    return (tn, fp, fn, tp), (FAR,FRR,EER)


def perform_testing(model, test_data, test_labels, threshold, seq_length = 30, sub_seq_matching = False):
    predictions = model.predict(test_data)
    if sub_seq_matching == True:
        predicted_classes = classify_prediction_huber(predictions,threshold,seq_length/2)
    else:
        predicted_classes = util.classify_predictions(predictions,threshold)
    (tn, fp, fn, tp), (FAR,FRR,EER) = calculate_keystroke_scores(test_labels,predicted_classes)
    return (predictions,predicted_classes),(tn, fp, fn, tp), (FAR,FRR,EER)

def classify_prediction_huber(predictions,threshold,ones_count_threshold):
    classified_predictions = []
    for i in range(len(predictions)):
        temp_class = util.classify_predictions(predictions[i],threshold)
        flattened_pred = temp_class.flatten()
        temp_counter = Counter(flattened_pred)
        if temp_counter[1]>ones_count_threshold:
            classified_predictions.append(1)
        else:
            classified_predictions.append(0)
    return np.array(classified_predictions).reshape(len(predictions),1)

def multi_class_testing(model, test_data, test_label, user_id):
    user_label = [0,0,0,0,0]
    user_index = participants.index(user_id)
    print("user index: {}".format(user_index))
    user_label[user_index] = 1
    print("user label: {}".format(user_label))

    y_true = []
    for label in test_label:
        if np.all(label == user_label):
            y_true.append(1)
        else:
            y_true.append(0)
    y_true = np.array(y_true).reshape(len(test_label),1)

    pred_proba = model.predict(test_data)
    pred_class = tf.argmax(pred_proba, axis = 1).numpy()

    y_pred = []
    for prediction in pred_class:
        if prediction == user_index:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred).reshape(len(test_label),1)
    (tn, fp, fn, tp), (FAR,FRR,EER) = calculate_keystroke_scores(y_true,y_pred)
    return (y_pred,y_true), (tn, fp, fn, tp), (FAR,FRR,EER)

def plot_results(history, title = None,plot_roc = False):
    plt.figure(figsize=(16, 4))
    epochs = len(history.history['val_loss'])
    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), history.history['loss'], label='Train Loss')
    plt.plot(range(epochs), history.history['val_loss'], label='Val Loss')
    plt.xticks(list(range(epochs)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if title!=None:
      plt.title(title)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), history.history['acc'], label='Train acc')
    plt.plot(range(epochs), history.history['val_acc'], label='Val Acc')
    plt.xticks(list(range(epochs)))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if title!=None:
      plt.title(title)
    plt.legend()
    
    if plot_roc==True:
        plt.subplot(1, 3, 3)
        plt.plot(range(epochs), history.history['auroc'], label='auroc')
        plt.plot(range(epochs), history.history['val_auroc'], label='Val auroc')
        plt.xticks(list(range(epochs)))
        plt.xlabel('Epochs')
        plt.ylabel('auroc')
        if title!=None:
          plt.title(title)
        plt.legend()

    return plt

