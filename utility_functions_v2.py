#from firebase import firebase
import tensorflow as tf
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout,LSTM,TimeDistributed,GRU
from tensorflow.keras.models import  Sequential, Model, load_model
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

#firebase = firebase.FirebaseApplication("https://key-stroke-dynamics-c823e.firebaseio.com/", None)


#this function fetches raw JSON data from db
#the JSON data contains basically 4 dict,
#keys_pressed dict - keys that were pressed in order
#key_pressed_timestamp dict - timestamp of each key press
#key_released dict - keys that were released in order
#key_release_timestamp dict - timestamp of each key release
def retrieve_json_data(db_string):
    user_id = "/key_stroke_data/"+db_string
    temp = firebase.get(user_id,None)
    return json.loads(temp)

#this function extract useful data from the JSON data
#main purpose is to find the keypress and key release timing pair of each key
#and return as [key,keypress_timestamp,key_released_timestamp]
def extract_timing_from_json_training(json_data,dict_name):
  key_pressed = dict_name + "_keys_pressed"
  key_released = dict_name + "_keys_released"
  key_pressed_timestamp = dict_name +"_key_pressed_timestamp"
  key_released_timestamp = dict_name +"_key_released_timestamp"
  temp_stack = []       #to store when press/release does not happens pairwise
  final_list = []     #final ans to return
  reverse_flag = False
  reverse_count = 0
  i = 0
  j = 0
  count = 0
  max_loop_count = len(json_data[key_released])
    
  while j<max_loop_count:
    #this if statement is for normal situation where first key was pressed and then released before next key is pressed
    if reverse_flag==False and (json_data[key_pressed][i] == json_data[key_released][j]):
      temp = [json_data[key_pressed][i], json_data[key_pressed_timestamp][i], json_data[key_released_timestamp][j]]
      final_list.append(temp)
    else:
      #if above situation does not occurs, check the stack first, size of stack will be 1 at any one time
      if len(temp_stack)!=0:
        if temp_stack[0][0]==json_data[key_pressed][i]:
          temp = [json_data[key_pressed][i], json_data[key_pressed_timestamp][i], temp_stack[0][1]]
          final_list.append(temp)
          temp_stack.pop()
      else:
        #this is for when user press shift to enter 1 capital letter
        #in this case the current release key is pushed to stack
        if reverse_flag==False and (json_data[key_pressed][i] == json_data[key_released][j+1]) :
          temp = [json_data[key_pressed][i], json_data[key_pressed_timestamp][i], json_data[key_released_timestamp][j+1]]
          final_list.append(temp)
          release_temp = [json_data[key_released][j],json_data[key_released_timestamp][j]]
          temp_stack.append(release_temp)
        else:
          #this is for when user press shift and enter multiple capital letters
          #it will search for the currently pressed key's release timing from release_dict
          if reverse_flag==False:
            temp_counter = j
            while True:
              if json_data[key_pressed][i]==json_data[key_released][temp_counter]:
                temp = [json_data[key_pressed][i], json_data[key_pressed_timestamp][i], json_data[key_released_timestamp][temp_counter]]
                final_list.append(temp)
                reverse_flag=True
                break
              reverse_count = reverse_count + 1
              temp_counter = temp_counter+1
            
          #press      release
          # shift     a    
          # a         b
          # b         shift
          #shift was found to be released afew keys after, so for subsequent loops,
          #check for press/release pairs -1 step
          else:
            if json_data[key_pressed][i] == json_data[key_released][j-1]:
              temp = [json_data[key_pressed][i], json_data[key_pressed_timestamp][i], json_data[key_released_timestamp][j-1]]
              final_list.append(temp)
              reverse_count = reverse_count -1
              if reverse_count ==0:
                reverse_flag = False

    i = i+1
    j = j+1
  return final_list

def extract_timing_from_json_testing(json_data):
  temp_stack = []       #to store when press/release does not happens pairwise
  another_list = []     #final ans to return
  reverse_flag = False
  reverse_count = 0
  i = 0
  j = 0
  count = 0
  not_found = False
  if len(json_data["key_released"])<len(json_data["key_pressed"]):
    max_loop_count = len(json_data["key_released"])
  elif len(json_data["key_released"])>len(json_data["key_pressed"]):
    max_loop_count = len(json_data["key_pressed"])
  else:
    max_loop_count = len(json_data["key_released"]) 
  while j<max_loop_count:
    #this if statement is for normal situation where first key was pressed and then released before next key is pressed
    if reverse_flag==False and (json_data["key_pressed"][i] == json_data["key_released"][j]):
      temp = [json_data["key_pressed"][i], json_data["key_pressed_timestamp"][i], json_data["key_released_timestamp"][j]]
      another_list.append(temp)
    else:
      #if above situation does not occurs, check the stack first, size of stack will be 1 at any one time
      if len(temp_stack)!=0:
        if temp_stack[0][0]==json_data["key_pressed"][i]:
          temp = [json_data["key_pressed"][i], json_data["key_pressed_timestamp"][i], temp_stack[0][1]]
          another_list.append(temp)
          temp_stack.pop()
      else:
        #this is for when user press shift to enter 1 capital letter
        #in this case the current release key is pushed to stack
        if reverse_flag==False and (json_data["key_pressed"][i] == json_data["key_released"][j+1]):
          temp = [json_data["key_pressed"][i], json_data["key_pressed_timestamp"][i], json_data["key_released_timestamp"][j+1]]
          another_list.append(temp)
          release_temp = [json_data["key_released"][j],json_data["key_released_timestamp"][j]]
          temp_stack.append(release_temp)
        else:
          #this is for when user press shift and enter multiple capital letters
          #it will search for the currently pressed key's release timing from release_dict
          if reverse_flag==False:
            temp_counter = j
            while True:
              if temp_counter == max_loop_count:
                print("not found!! popping from list")
                json_data["key_pressed"].pop(i)
                reverse_count = 0
                break
              if json_data["key_pressed"][i]==json_data["key_released"][temp_counter]:
                temp = [json_data["key_pressed"][i], json_data["key_pressed_timestamp"][i], json_data["key_released_timestamp"][temp_counter]]
                another_list.append(temp)
                reverse_flag=True
                break
              reverse_count = reverse_count + 1
              temp_counter = temp_counter+1
          #press      release
          # shift     a    
          # a         b
          # b         shift
          #shift was found to be released afew keys after, so for subsequent loops,
          #check for press/release pairs -1 step
          else:
            if json_data["key_pressed"][i] == json_data["key_released"][j-1]:
              temp = [json_data["key_pressed"][i], json_data["key_pressed_timestamp"][i], json_data["key_released_timestamp"][j-1]]
              another_list.append(temp)
              reverse_count = reverse_count -1
              if reverse_count ==0:
                reverse_flag = False

    i = i+1
    j = j+1
  return another_list


  
#calculate keystroke information of a given key
#returns [keypress_timestamp,hold_time,latency,speed,up_up]
#calculate keystroke information of a given key
#returns [key_press_index,hold_time,latency,speed,up_up]
def vectors_calculations(index,keypress_1,keyrelease_1,keypress_2,keyrelease_2):
  
  hold_time = keyrelease_1 - keypress_1

  #last key reached
  if keypress_2==0 or keyrelease_2==0:
    return (index,hold_time,0,0)
  
  latency = keypress_2 - keyrelease_1
  speed = keypress_2 - keypress_1 
  return (index,hold_time,latency,speed)

#calls vectors_calculation function multiple times 
#to calculate keystroke timings of every single key
def convert_to_vec(timing_data):
  vector_seq = []
  length = len(timing_data)
  for i in range(length):
    if i==length-1:
      temp = vectors_calculations(i,timing_data[i][1],timing_data[i][2],0,0)
    else:
      temp = vectors_calculations(i,timing_data[i][1],timing_data[i][2],timing_data[i+1][1],timing_data[i+1][2])
    vector_seq.append(np.array(temp))
  return vector_seq

#this function take each keystroke information and line them up to seq_length
#[[key1_timestamp,holdtime,latency,speed,up_up],[key2_timestamp,holdtime,latency,speed,up_up]......]
#the purpose of doing this is to feed data based on timestamp into the timedistributed cnn layer
#basically a sliding window in time step
def vec_to_seq(vector_data,seq_length):
  temp_shape = np.array(vector_data).shape[-1]
  output_seq = []
  index = 0
  curr_first_index = 0
  while True:
    temp = []
    for inner_loop_counter in range(seq_length):
      index = curr_first_index + inner_loop_counter
      temp.append(vector_data[index])
    temp_arr = np.array(temp).reshape(seq_length,temp_shape)
    output_seq.append(temp_arr)
    curr_first_index +=1
    if (curr_first_index+seq_length)> len(vector_data):
      break
  return output_seq


def get_vectors_for_1_session(json_link):
  json_data = retrieve_json_data(json_link)
  name_list = ["obama","your_day","wuhan"]
  timing_data = []
  #timing_data contains[[obama],[your_day],[wuhan]]
  #in terms of [key,keypress,keyrelease]
  for name in name_list:
    temp = extract_timing_from_json_training(json_data,name)
    timing_data.append(temp)
  
  #vec_list contains [[obama],[your_day],[wuhan]]
  vec_list = []
  for i in range(len(timing_data)):
    curr_vec = convert_to_vec(timing_data[i])
    vec_list.append(curr_vec)
  return vec_list

def flatten_vector(vector):
    original_length = [len(vector[0]),len(vector[1]),len(vector[2])]
    flattened_vec = []
    for i in range(len(vector)):
        if flattened_vec == None:
            flattened_vec = vector[i]
        else: 
            flattened_vec.extend(vector[i])
    return flattened_vec,original_length


def restore_from_flattened_1_session(flattened_vec,length_list):
    original_vec = []
    curr_index = 0
    for i in range(len(length_list)):
        upper_bound = curr_index + length_list[i]
        temp =  flattened_vec[curr_index:upper_bound]
        original_vec.append(temp)
        curr_index = curr_index + length_list[i]
    return original_vec   


def get_seq_for_1_session(vector_list,seq_length):
  combined_seq = []
  for i in range(len(vector_list)):
    curr_seq = vec_to_seq(vector_list[i],seq_length)
    if combined_seq == None:
      combined_seq.append(curr_seq)
    else:
      combined_seq.extend(curr_seq)
  return combined_seq

def get_vectors_for_multi_sessions(json_links):
    combined_vectors = []
    for link in json_links:
        temp = get_vectors_for_1_session(link)
        combined_vectors.append(temp)
    return combined_vectors
            
def flatten_multi_vec(combined_vec):
    length_list = []
    combined_flatten_vec = []
    for i in range(len(combined_vec)):
        temp_vec,temp_length_list = flatten_vector(combined_vec[i])
        combined_flatten_vec.extend(temp_vec)
        length_list.append(temp_length_list)
    return combined_flatten_vec,length_list

def scale_data(train_data):
    print("Using MinMaxScaler")
    scaler = MinMaxScaler((-1,1))
    scaler.fit(train_data)
    scaled_data = scaler.transform(train_data)
    return (scaled_data,scaler)
        
def restore_from_flattened_vec_multi_session(flattened_vec,length_list):
    original_vec = []
    curr_index = 0
    for i in range(len(length_list)):
        vec_per_db = []
        for j in range(len(length_list[i])):
            upper_bound = curr_index + length_list[i][j]
            temp =  flattened_vec[curr_index:upper_bound]
            vec_per_db.append(temp)
            curr_index = curr_index + length_list[i][j]
        original_vec.append(vec_per_db)
    return original_vec       

def convert_to_seq_multi_session(vector_list,seq_length):
    seq = []
    for i in range(len(vector_list)):
        temp_seq = get_seq_for_1_session(vector_list[i],seq_length)
        seq.extend(temp_seq)
    return seq    

def make_label(length,label_type):
    if label_type==1:
        label = np.ones((length,1))
    else:
        label = np.zeros((length,1))
    return label

def combine_data_from_sessions(user_db_list_train,user_test):
  #data from multiple sessions are combined to make training data
  train_data = []
  test_data = generate_data(user_test)
  for db in user_db_list_train:
    curr_vec_seq = generate_data(db)
    print("from {}, extracted vec seq: {}".format(db,np.array(curr_vec_seq).shape))
    if train_data==None:
      train_data.append(curr_vec_seq)
    else:
      train_data.extend(curr_vec_seq)
  scaled_train_data,scaled_test_data,scaler = scale_data(train_data,test_data)
  return scaled_train_data,scaled_test_data,scaler

def build_model(rnn_type,rnn_state,num_conv_filters):
      #input shape (batch_size,time_axis,size,size)
      model = Sequential()
      model.add(TimeDistributed(Conv1D(num_conv_filters, 2, activation='relu'), input_shape=(None,5,1)))
      model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
      model.add(TimeDistributed(Flatten()))
      model.add(rnn_type(50,stateful = rnn_state,return_sequences=True))
      model.add(Dropout(0.3))
      model.add(rnn_type(50,stateful = rnn_state,return_sequences=True))
      model.add(Dropout(0.3))
      model.add(rnn_type(50,stateful = rnn_state,dropout=0.5))
      model.add(Dense(1, activation='sigmoid'))
        
      return model

#tuning optimiser used
def grid_search_optmiser(data,labels,lr):
  optimisers = {"SGD":      tf.keras.optimizers.SGD(learning_rate=lr),
                "RMSprop":  tf.keras.optimizers.RMSprop(learning_rate=lr),
                "Adam":     tf.keras.optimizers.Adam(learning_rate=lr),
                "Adadelta": tf.keras.optimizers.Adadelta(learning_rate=lr),
                "Adagrad":  tf.keras.optimizers.Adagrad(learning_rate=lr),
                "Adamax":   tf.keras.optimizers.Adamax(learning_rate=lr),
                "Nadam":    tf.keras.optimizers.Nadam(learning_rate=lr),
                "Ftrl":     tf.keras.optimizers.Ftrl(learning_rate=lr)}
  for key,value in optimisers.items():
    model = build_model(LSTM,False,5)
    model.compile(optimizer=value,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(data, labels, epochs=100, batch_size=32,validation_split=0.2,shuffle=True,verbose=0)
    plot_results(history,key).show()
    
    
def plot_results(history, title = None):
    plt.figure(figsize=(8, 3))
    epochs = len(history.history['val_loss'])
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history.history['loss'], label='Train Loss')
    plt.plot(range(epochs), history.history['val_loss'], label='Val Loss')
    plt.xticks(list(range(epochs)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if title!=None:
      plt.title(title)

    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history.history['acc'], label='Acc')
    plt.plot(range(epochs), history.history['val_acc'], label='Val Acc')
    plt.xticks(list(range(epochs)))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if title!=None:
      plt.title(title)
    plt.legend()
    return plt

def generate_test_data(user_data,intruder_data):
  test_data = np.vstack((user_data,intruder_data))
  pos_label = np.ones((user_data.shape[0],1))
  neg_label = np.zeros((intruder_data.shape[0],1))
  test_label = np.vstack((pos_label,neg_label))
  return (test_data,test_label)

def classify_predictions(predictions,threshold):
  predictions = np.where(np.greater(predictions,threshold),1,0)
  return predictions

def calculate_FPR_FRR(TN, FP, FN, TP):
  FAR = FP/(FP + TN)
  FRR = FN/(FN + TP)
  return FAR,FRR
  
def calculate_eer(y_true,y_score):
  from scipy.optimize import brentq
  from scipy.interpolate import interp1d
  from sklearn.metrics import roc_curve

  FAR, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
  FRR = 1 - tpr
  EER = brentq(lambda x : 1. - x - interp1d(FAR, tpr)(x), 0., 1.)
  return EER




