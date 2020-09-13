import codecs
import utility_functions_v2 as util
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout,LSTM,TimeDistributed,GRU  
from tensorflow.keras.layers import Bidirectional,BatchNormalization,Activation,Concatenate, Multiply, Subtract, Lambda
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from collections import Counter
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score

weight_decay = 1e-4

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


def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def cosine_distance(vests):
    x, y = vests
    x = tf.math.l2_normalize(x, axis=-1)
    y = tf.math.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def sum_Manhattan_distance(left, right):
    return K.sum(K.abs(left-right),axis=1,keepdims=True)

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def abs_diff(left,right):
    return K.abs(left-right)
   
def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

def custom_loss(y_true,y_pred,clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)
    
def get_call_backs():
    early_stopping = EarlyStopping(monitor = 'val_loss',patience = 30, mode = "min")
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=15)
    return (early_stopping, learning_rate_reduction)

def get_call_backs_2():
    early_stopping = EarlyStopping(monitor = 'val_loss',patience = 30, mode = "min")
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=20, min_lr=0.001)
    return (early_stopping, learning_rate_reduction)


def build_cnn_lstm_siamese_model_cos_dist(input_shape,drop_out, num_cnn = 32, num_lstm = 64):
    print("Using cosine dist - 64 CNN filters, 80 LSTM !!!")
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    #input shape (batch_size,time_axis,size,size)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_cnn, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay),activation = "relu",dropout = drop_out))
    
    model.add(BatchNormalization())
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    x3 = Subtract()([encoded_l, encoded_r])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([encoded_l, encoded_l])
    x2_ = Multiply()([encoded_r, encoded_r])
    x4 = Subtract()([x1_, x2_])
    
    #https://stackoverflow.com/a/51003359/10650182
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([encoded_l, encoded_r])
    
    conc = Concatenate(axis=-1)([x5,x4, x3])
    x = Dense(100, activation="relu", name='conc_layer')(conc)
    x = Dropout(0.3)(x)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(x)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    return siamese_net

def build_cnn_lstm_siamese_model_cos_dist_ver_2(input_shape, drop_out, num_cnn = 32, num_lstm = 64):
    print("Using cosine dist configuration 2 - {} CNN filters, {} LSTM !!!!".format(num_cnn,num_lstm))
      
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    #input shape (batch_size,time_axis,size,size)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_cnn, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(LSTM(num_lstm, activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, activation = "relu",dropout = drop_out))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    x3 = Subtract()([encoded_l, encoded_r])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([encoded_l, encoded_l])
    x2_ = Multiply()([encoded_r, encoded_r])
    x4 = Subtract()([x1_, x2_])
    
    #https://stackoverflow.com/a/51003359/10650182
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([encoded_l, encoded_r])
    
    conc = Concatenate(axis=-1)([x5,x4, x3])
    x = Dense(100, activation="relu", name='conc_layer')(conc)
    x = Dropout(0.3)(x)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(x)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    return siamese_net

def build_cnn_lstm_siamese_model_cos_dist_ver_3(input_shape, drop_out, num_cnn = 32, num_lstm = 64):
    print("Using cosine dist configuration 3 - {} CNN filters, {} LSTM !!!!".format(num_cnn,num_lstm))
      
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    #input shape (batch_size,time_axis,size,size)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_cnn, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(LSTM(num_lstm,  kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm,  kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm,  kernel_regularizer=l2(weight_decay), activation = "relu",dropout = drop_out))
    
    
    model.add(Flatten())
    model.add(BatchNormalization())
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    x3 = Subtract()([encoded_l, encoded_r])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([encoded_l, encoded_l])
    x2_ = Multiply()([encoded_r, encoded_r])
    x4 = Subtract()([x1_, x2_])
    
    #https://stackoverflow.com/a/51003359/10650182
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([encoded_l, encoded_r])
    
    conc = Concatenate(axis=-1)([x5,x4, x3])
    x = Dense(100, activation="relu", name='conc_layer')(conc)
    x = Dropout(0.3)(x)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(x)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    return siamese_net

def build_cnn_lstm_siamese_model_var_dist(input_shape, similarity_measure, drop_out, num_cnn = 32, num_lstm = 64): 
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    #input shape (batch_size,time_axis,size,size)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_cnn, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    
    model.add(TimeDistributed(Flatten()))
        
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",dropout = drop_out))
    
    model.add(BatchNormalization())
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Calculates the distance as defined by the MaLSTM model
    #malstm = Lambda(sum_Manhattan_distance, output_shape =manhanttan_output_shapes)([encoded_l,encoded_r])
    if similarity_measure=="malstm":
        print("Using Manhattan distance configuration 1!!!")
        custom_similarity_layer = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([encoded_l, encoded_r])
    elif similarity_measure=="abs_diff":
        print("Using absolute difference configuration 1!!!")
        custom_similarity_layer = Lambda(function=lambda x: sum_Manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([encoded_l, encoded_r])
        #custom_similarity_layer = Lambda(function = lambda x:  abs_diff(x[0],x[1]))([encoded_l, encoded_r])

    x = Dense(100, activation="relu", name='conc_layer')(custom_similarity_layer)
    x = Dropout(0.3)(x)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(x)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    return siamese_net

def build_cnn_lstm_siamese_model_var_dist_ver_2(input_shape, similarity_measure, drop_out, num_cnn = 32, num_lstm = 64):
      
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    #input shape (batch_size,time_axis,size,size)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_cnn, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
        
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Calculates the distance as defined by the MaLSTM model
    #malstm = Lambda(sum_Manhattan_distance, output_shape =manhanttan_output_shapes)([encoded_l,encoded_r])
    if similarity_measure=="malstm":
        print("Using Manhattan distance configuration 2!!!")
        custom_similarity_layer = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([encoded_l, encoded_r])
    elif similarity_measure=="abs_diff":
        print("Using absolute difference configuration 2!!!")
        custom_similarity_layer = Lambda(function = lambda x:  abs_diff(x[0],x[1]))([encoded_l, encoded_r])

    x = Dense(100, activation="relu", name='conc_layer')(custom_similarity_layer)
    x = Dropout(0.3)(x)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(x)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    return siamese_net

def build_cnn_lstm_siamese_model_var_dist_ver_3(input_shape, similarity_measure, drop_out, num_cnn = 32, num_lstm = 64):
      
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    #input shape (batch_size,time_axis,size,size)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(num_cnn, 2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    
    model.add(TimeDistributed(Flatten()))
        
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",return_sequences=True,dropout = drop_out))
    model.add(LSTM(num_lstm, kernel_regularizer=l2(weight_decay), activation = "relu",dropout = drop_out))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Calculates the distance as defined by the MaLSTM model
    #malstm = Lambda(sum_Manhattan_distance, output_shape =manhanttan_output_shapes)([encoded_l,encoded_r])
    if similarity_measure=="malstm":
        print("Using Manhattan distance configuration 3!!!")
        custom_similarity_layer = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([encoded_l, encoded_r])
    elif similarity_measure=="abs_diff":
        print("Using absolute difference configuration 3!!!")
        custom_similarity_layer = Lambda(function = lambda x:  abs_diff(x[0],x[1]))([encoded_l, encoded_r])

    x = Dense(100, activation="relu", name='conc_layer')(custom_similarity_layer)
    x = Dropout(0.1)(x)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(x)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    return siamese_net



def train_siamese_model(model,optimiser,loss_func,callbacks,metrics,left_data_train, right_data_train,train_label,left_data_val, right_data_val,val_label,epochs,batch_size):
    print("loss function used: {}".format(loss_func))
    model.compile(loss=loss_func, metrics=metrics, optimizer=optimiser)
    history = model.fit([left_data_train, right_data_train],train_label,
                        epochs=epochs,batch_size = batch_size,
                        callbacks = callbacks,
                        validation_data=([left_data_val, right_data_val],val_label))
    if metrics==["acc"]:
        plot_results(history).show()
    else:
        plot_results(history,None,True).show()
    return history


def calculate_keystroke_scores(y_true,y_pred_class):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    FAR,FRR = util.calculate_FPR_FRR(tn, fp, fn, tp)
    EER = util.calculate_eer(y_true,y_pred_class)
    return (tn, fp, fn, tp), (FAR,FRR,EER)


def siamese_testing(predictions,threshold,y_true):
    predicted_classes = util.classify_predictions(predictions,threshold)
    (tn, fp, fn, tp), (FAR,FRR,EER) = calculate_keystroke_scores(y_true,predicted_classes)
    return predicted_classes,(tn, fp, fn, tp), (FAR,FRR,EER)

def print_model_scores(model_list,test_label,left_test_data,right_test_data,threshold):
    scores = []
    for i in range(len(model_list)):
        curr_pred = model_list[i].predict([left_test_data,right_test_data])
        curr_pred_classes, (tn, fp, fn, tp), (FAR,FRR,EER) = siamese_testing(curr_pred,threshold,test_label)
        print("TN:    {},    FP:   {},   FN:   {},   TP:   {}".format(tn, fp, fn, tp))
        print("FAR:   {},   FRR:   {},   EER:   {}".format(FAR, FRR, EER))
        print("")
        scores.append([curr_pred_classes, (tn, fp, fn, tp), (FAR, FRR, EER)])
    return scores