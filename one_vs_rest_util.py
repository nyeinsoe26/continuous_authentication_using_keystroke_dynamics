import codecs
import utility_functions_v2 as util
import load_buffalo_data as data_loader
import cnn_lstm_models as models
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
from random import sample

user_id = "005"
intruder_id_list = ["080","083","017","019"]


#declare empty dict
def make_empty_keystroke_dict():
    my_keystroke_dict = {
        "key_pressed" : [],
        "key_released": [],
        "key_pressed_timestamp": [],
        "key_released_timestamp":[]
    }
    return my_keystroke_dict


def read_data_for_user(user_id,sess_num,key_board_type,task,db):
    file_name = user_id + sess_num + key_board_type + task
    path_string = r"C:\Users\nyein\Intership_proj\read_data\UB_keystroke_dataset\s" + sess_num+ "\\baseline\\" + file_name + ".txt"
    delimiter = ' '
    reader = codecs.open(path_string, 'r', encoding='utf-8')
    for line in reader:
        row = line.split(delimiter)
        if row[1]=="KeyDown":
            db["key_pressed"].append(row[0])
            db["key_pressed_timestamp"].append(np.float64(row[2]))
        if row[1]=="KeyUp":
            db["key_released"].append(row[0])
            db["key_released_timestamp"].append(np.float64(row[2]))
    return db

def read_data_for_intruder(intruder_id,sess_num,key_board_type,task,db):
    file_name = intruder_id + sess_num + key_board_type + task
    path_string = r"C:\Users\nyein\Intership_proj\read_data\UB_keystroke_dataset\s" + sess_num+ "\\rotation\\" + file_name + ".txt"
    delimiter = ' '
    reader = codecs.open(path_string, 'r', encoding='utf-8')
    for line in reader:
        row = line.split(delimiter)
        if row[1]=="KeyDown":
            db["key_pressed"].append(row[0])
            db["key_pressed_timestamp"].append(np.float64(row[2]))
        if row[1]=="KeyUp":
            db["key_released"].append(row[0])
            db["key_released_timestamp"].append(np.float64(row[2]))
    return db

#this function is to help with training data generation, else it serves no purpose
def read_4_parts_data(id):
    if int(id)>=80:
        part1 = read_data_for_intruder(id,"0","1","0",make_empty_keystroke_dict())
        part2 = read_data_for_intruder(id,"0","1","1",make_empty_keystroke_dict())
        part3 = read_data_for_intruder(id,"1","2","1",make_empty_keystroke_dict())
        part4 = read_data_for_intruder(id,"1","2","0",make_empty_keystroke_dict())
    else:
        part1 = read_data_for_user(id,"0","0","0",make_empty_keystroke_dict())
        part2 = read_data_for_user(id,"0","0","1",make_empty_keystroke_dict())
        part3 = read_data_for_user(id,"1","0","0",make_empty_keystroke_dict())
        part4 = read_data_for_user(id,"1","0","1",make_empty_keystroke_dict())

    return part1,part2,part3,part4

def convert_to_keystroke_vec(db):
    timing_vec = util.extract_timing_from_json_testing(db)
    keystroke_vec = util.convert_to_vec(timing_vec)
    return keystroke_vec

#this function is to help with training data generation, else it serves no purpose
def convert_4parts_data_to_keystroke_vec(db1,db2,db3,db4):
    keystroke_vec_1 = convert_to_keystroke_vec(db1)
    keystroke_vec_2 = convert_to_keystroke_vec(db2)
    keystroke_vec_3 = convert_to_keystroke_vec(db3)
    keystroke_vec_4 = convert_to_keystroke_vec(db4)
    return keystroke_vec_1, keystroke_vec_2, keystroke_vec_3, keystroke_vec_4


def get_keystroke_seq(id, seq_length=30, return_length = False):
    part1,part2,part3,part4 = read_4_parts_data(id)
    keystroke_vec_1, keystroke_vec_2, keystroke_vec_3, keystroke_vec_4 = convert_4parts_data_to_keystroke_vec(part1,part2,part3,part4)
   
    keystroke_seq_1 = util.vec_to_seq(keystroke_vec_1,seq_length)
    keystroke_seq_2 = util.vec_to_seq(keystroke_vec_2,seq_length)
    keystroke_seq_3 = util.vec_to_seq(keystroke_vec_3,seq_length)
    keystroke_seq_4 = util.vec_to_seq(keystroke_vec_4,seq_length)

    train_data = keystroke_seq_1 + keystroke_seq_2 + keystroke_seq_3 + keystroke_seq_4
    if return_length==False:
        return train_data
    else:
        length_list = [len(keystroke_seq_1), len(keystroke_seq_2), len(keystroke_seq_3), len(keystroke_seq_4)]
        return train_data, length_list
    
    




def scale_data(train_data):
    #print("Using standard scaler")
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_data = scaler.transform(train_data)
    return (scaled_data,scaler)

#this function is used when generating test data
#this only loads 1 session of data
def get_seq_data(db,scaler,seq_length, skip_scaling = False):
    db_timing_vec =  util.extract_timing_from_json_testing(db)
    db_keystroke_vec = util.convert_to_vec(db_timing_vec)
    if skip_scaling !=True:
        db_keystroke_vec = scaler.transform(db_keystroke_vec)
    db_seq_data = util.vec_to_seq(db_keystroke_vec,seq_length)
    return db_seq_data

#some helper functions
def make_labels(pos_data,neg_data,sub_string_cross_entropy=False,seq_length=30):
    if sub_string_cross_entropy==False:
        pos_label = util.make_label(len(pos_data),1)
        neg_label = util.make_label(len(neg_data),0)
    else:
        pos_label = np.ones((len(pos_data),seq_length,1))
        neg_label = np.zeros((len(neg_data),seq_length,1))
    return pos_label,neg_label

def prep_data(pos_data,neg_data,n_dim,sub_string_cross_entropy=False):
    pos_labels,neg_labels = make_labels(pos_data,neg_data,sub_string_cross_entropy)
    test_data = np.vstack((pos_data,neg_data))
    if n_dim==4:
        test_data = np.expand_dims(test_data, axis=-1)
    test_labels = np.vstack((pos_labels,neg_labels))
    return test_data,test_labels

def calculate_keystroke_scores(y_true,y_pred_class):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    FAR,FRR = util.calculate_FPR_FRR(tn, fp, fn, tp)
    EER = util.calculate_eer(y_true,y_pred_class)
    print("TN:    {},    FP:   {},   FN:   {},   TP:   {}".format(tn, fp, fn, tp))
    print("FAR:   {},   FRR:   {},   EER:   {}".format(FAR, FRR, EER))
    return (tn, fp, fn, tp), (FAR,FRR,EER)

def load_train_data(user_id,intruder_id, add_noise = False, seq_length=30, num_features = 4):

    #load user_seq data
    user_seq_data = get_keystroke_seq(user_id,seq_length)
    user_data_length = len(user_seq_data)

    #load intruder seq data
    intruder_seq_data = get_keystroke_seq(intruder_id,seq_length)
    intruder_data_length = len(intruder_seq_data)

    #adding noise to secure data
    if add_noise==True:
        print("loading noise data")
        noise = generate_gaussian_noise(intruder_seq_data.shape,0.1)
        intruder_seq_data = intruder_seq_data + noise

    #combine all data
    train_data = np.vstack((user_seq_data,intruder_seq_data))
    train_data = train_data.reshape(len(train_data),seq_length*num_features)

    #scale data
    total_length = user_data_length + intruder_data_length
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_data = scaler.transform(train_data)
    scaled_data = scaled_data.reshape(total_length,seq_length,num_features)
    scaled_data = np.expand_dims(scaled_data, axis=-1)

    #make label
    pos_label = np.ones((user_data_length,1))
    neg_label = np.zeros((intruder_data_length,1))
    train_label = np.vstack((pos_label,neg_label))

    for i in range(10):
        scaled_data, train_label = shuffle(scaled_data, train_label)
    print("Train data shape: {}, Train label shape: {}".format(scaled_data.shape, train_label.shape))
    return scaled_data,train_label,scaler

def read_1_session_data(id):
    if int(id)>=80:
        db = read_data_for_intruder(id,"2","3","0",make_empty_keystroke_dict())
    else:
        db = read_data_for_user(id,"2","0","0",make_empty_keystroke_dict())
    return db

def load_test_data(user_id, intru_id, scaler, seq_length, num_features = 4):
    #load user test data
    db_5 = read_1_session_data(user_id)
    db_5_seq_data = get_seq_data(db_5,scaler, seq_length, True)
    db_7 = read_1_session_data(intru_id)
    db_7_seq_data = get_seq_data(db_7,scaler, seq_length, True)
    
    from random import sample
    pos_data = sample(db_5_seq_data,500)
    neg_data = sample(db_7_seq_data,500)

    test_data = np.vstack((pos_data,neg_data))
    test_data = test_data.reshape(len(test_data), seq_length*num_features)

    scaled_test_data = scaler.transform(test_data)
    scaled_test_data = scaled_test_data.reshape(len(test_data),seq_length,num_features)
    scaled_test_data = np.expand_dims(scaled_test_data, axis=-1)

    pos_label = np.ones((len(pos_data),1))
    neg_label = np.zeros((len(neg_data),1))
    test_label = np.vstack((pos_label,neg_label))
    print("test_data shape: {}, test_label shape: {}".format(scaled_test_data.shape,test_label.shape))
    return scaled_test_data,test_label

def load_one_vs_one_train_data(user_id,intruder_id, add_noise = False, seq_length=30, num_features = 4):
    _,__,global_scaler = load_multi_class_train_data(user_id, intruder_id_list, seq_length, add_noise , num_features)
    #load user_seq data
    user_seq_data = get_keystroke_seq(user_id,seq_length)
    user_data_length = len(user_seq_data)

    #load intruder seq data
    intruder_seq_data = get_keystroke_seq(intruder_id,seq_length)
    intruder_data_length = len(intruder_seq_data)

    #adding noise to secure data
    if add_noise==True:
        print("loading noise data")
        noise = generate_gaussian_noise(intruder_seq_data.shape,0.1)
        intruder_seq_data = intruder_seq_data + noise

    #combine all data
    train_data = np.vstack((user_seq_data,intruder_seq_data))
    train_data = train_data.reshape(len(train_data),seq_length*num_features)

    #scale data
    total_length = user_data_length + intruder_data_length
    scaled_data = global_scaler.transform(train_data)
    scaled_data = scaled_data.reshape(total_length,seq_length,num_features)
    scaled_data = np.expand_dims(scaled_data, axis=-1)

    #make label
    pos_label = np.ones((user_data_length,1))
    neg_label = np.zeros((intruder_data_length,1))
    train_label = np.vstack((pos_label,neg_label))

    for i in range(10):
        scaled_data, train_label = shuffle(scaled_data, train_label)
    print("Train data shape: {}, Train label shape: {}".format(scaled_data.shape, train_label.shape))
    return scaled_data,train_label,global_scaler

def perform_one_vs_one_testing(model_list,test_data,y_true):
    print("one vs one testing ###!")
    y_pred = []
    for data in test_data:
        curr_pred = []
        for model in model_list:
            temp = model.predict(np.array([data]))
            temp = util.classify_predictions(temp,0.5)
            curr_pred.append(temp[0][0])
        if list_item_counters(curr_pred,1) >1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    (tn, fp, fn, tp), (FAR,FRR,EER) = models.calculate_keystroke_scores(y_true,y_pred)
    return (tn, fp, fn, tp), (FAR,FRR,EER)

def list_item_counters(input_list,item):
    #print("input list: {}".format(input_list))
    from collections import Counter
    temp_dict = Counter(input_list)
    return temp_dict[item]



def generate_gaussian_noise(noise_shape,sd):
    noise = np.random.normal(0, sd, noise_shape)
    return noise


#this function loads 5 people's data at once for multiclass classification
#each user is assigned label 0-4
def load_multi_class_train_data(user_id, intruder_id_list, seq_length, add_noise = False, num_features = 4):
    
    #load raw user data
    user_seq_data = get_keystroke_seq(user_id,seq_length)
    user_data_length = len(user_seq_data)

    intruder_seq_data = []
    intruder_data_length = []
    for intruder in intruder_id_list:
        curr_intruder_test_seq = get_keystroke_seq(intruder, seq_length)
        intruder_seq_data.append(curr_intruder_test_seq)
        intruder_data_length.append(len(curr_intruder_test_seq))

    intruder_seq_data = np.vstack((intruder_seq_data))
    #adding noise to secure data
    if add_noise==True:
        print("loading noise data")
        noise = generate_gaussian_noise(intruder_seq_data.shape,0.1)
        intruder_seq_data = intruder_seq_data + noise
    #combine all data
    train_data = np.vstack((user_seq_data,intruder_seq_data))
    train_data = train_data.reshape(len(train_data),seq_length*num_features)

    #scale data
    total_length = user_data_length + sum(intruder_data_length)
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_data = scaler.transform(train_data)
    scaled_data = scaled_data.reshape(total_length,seq_length,num_features)
    scaled_data = np.expand_dims(scaled_data, axis=-1)

    #generate multiclass labels
    train_label = create_multi_class_labels(user_data_length, intruder_data_length)

    for i in range(10):
        scaled_data, train_label = shuffle(scaled_data, train_label)
    print("Multi-class Train data shape: {}, Multi-class Train label shape: {}".format(scaled_data.shape, train_label.shape))
    return scaled_data,train_label,scaler

def create_multi_class_labels(user_length, intruder_length_list):
    user_label = np.zeros((user_length,1))
    
    intruder_labels = []
    i = 0
    while i<len(intruder_length_list):
        curr_intruder_label = np.ones((intruder_length_list[i],1))*(i+1)
        intruder_labels.append(curr_intruder_label)
        i +=1
    intruder_labels = np.vstack((intruder_labels))
    labels = np.vstack((user_label,intruder_labels))
    labels = tf.keras.utils.to_categorical(labels)
    return labels


def load_multi_class_test_data(user_id, intruder_id_list, scaler, seq_length, add_noise = False, num_features=4):
    #load user test data
    user_test_db = read_1_session_data(user_id)
    user_test_seq = get_seq_data(user_test_db,scaler,seq_length, True)
    user_test_seq = sample(user_test_seq,500)
    user_data_length = len(user_test_seq)

    #load intruder test data
    intruder_test_seq = []
    intruder_data_length = []
    for intruder in intruder_id_list:
        curr_intruder_db = read_1_session_data(intruder)
        curr_intruder_test_seq = get_seq_data(curr_intruder_db, scaler, seq_length, True)
        curr_intruder_test_seq = sample(curr_intruder_test_seq,500)
        intruder_test_seq.append(curr_intruder_test_seq)
        intruder_data_length.append(len(curr_intruder_test_seq))

    #creating test data
    intruder_test_seq = np.vstack((intruder_test_seq))

    #adding noise to secure data
    if add_noise==True:
        print("loading noise data")
        noise = generate_gaussian_noise(intruder_test_seq.shape,0.1)
        intruder_test_seq = intruder_test_seq + noise

    test_data = np.vstack((user_test_seq,intruder_test_seq))
    test_data = test_data.reshape(len(test_data),seq_length*num_features)
    scaled_test_data = scaler.transform(test_data)
    scaled_test_data = scaled_test_data.reshape(len(test_data),seq_length,num_features)
    scaled_test_data = np.expand_dims(scaled_test_data, axis=-1)

    test_label = create_multi_class_labels(user_data_length, intruder_data_length)
    print("Multi-class Test data shape: {}, Multi-class Test label shape: {}".format(scaled_test_data.shape,test_label.shape))
    return scaled_test_data,test_label


def load_one_vs_rest_train_data(user_id, intruder_id_list, seq_length, add_noise = False, num_features = 4):
    #load raw user data
    user_seq_data = get_keystroke_seq(user_id,seq_length)
    user_data_length = len(user_seq_data)

    #load intruder test data
    intruder_seq_data = []
    intruder_data_length = []
    for intruder in intruder_id_list:
        curr_intruder_test_seq = get_keystroke_seq(intruder, seq_length)
        temp = sample(curr_intruder_test_seq, user_data_length//len(intruder_id_list))
        intruder_seq_data.append(temp)
        intruder_data_length.append(len(temp))

    intruder_seq_data = np.vstack((intruder_seq_data))
    #adding noise to secure data
    if add_noise==True:
        print("loading noise data")
        noise = generate_gaussian_noise(intruder_seq_data.shape,0.1)
        intruder_seq_data = intruder_seq_data + noise

    #combine all data
    train_data = np.vstack((user_seq_data,intruder_seq_data))
    train_data = train_data.reshape(len(train_data), seq_length*num_features)

    #scale data
    total_length = user_data_length + sum(intruder_data_length)
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_data = scaler.transform(train_data)
    scaled_data = scaled_data.reshape(total_length,seq_length,num_features)
    scaled_data = np.expand_dims(scaled_data, axis=-1)

    #generate label
    pos_label = np.ones((user_data_length,1))
    neg_label = np.zeros((sum(intruder_data_length),1))
    train_label = np.vstack((pos_label,neg_label))

    for i in range(10):
        scaled_data, train_label = shuffle(scaled_data, train_label)
    print("Train data shape: {}, Train label shape: {}".format(scaled_data.shape, train_label.shape))
    return scaled_data,train_label,scaler

def load_one_vs_rest_test_data(user_id, intruder_id_list, scaler, seq_length, add_noise = False, num_features=4):
    #load user test data
    user_test_db = read_1_session_data(user_id)
    user_test_seq = get_seq_data(user_test_db,scaler,seq_length, True)
    user_test_seq = sample(user_test_seq,600)
    user_data_length = len(user_test_seq)

    #load intruder test data
    intruder_test_seq = []
    intruder_data_length = []
    for intruder in intruder_id_list:
        curr_intruder_db = read_1_session_data(intruder)
        curr_intruder_test_seq = get_seq_data(curr_intruder_db, scaler, seq_length, True)
        curr_intruder_test_seq = sample(curr_intruder_test_seq,150)
        intruder_test_seq.append(curr_intruder_test_seq)
        intruder_data_length.append(len(curr_intruder_test_seq))

    #creating test data
    intruder_test_seq = np.vstack((intruder_test_seq))

    #adding noise to secure data
    if add_noise==True:
        print("loading noise data")
        noise = generate_gaussian_noise(intruder_test_seq.shape,0.1)
        intruder_test_seq = intruder_test_seq + noise

    test_data = np.vstack((user_test_seq,intruder_test_seq))
    test_data = test_data.reshape(len(test_data),seq_length*num_features)
    scaled_test_data = scaler.transform(test_data)
    scaled_test_data = scaled_test_data.reshape(len(test_data),seq_length,num_features)
    scaled_test_data = np.expand_dims(scaled_test_data, axis=-1)

    #generate label
    pos_label = np.ones((user_data_length,1))
    neg_label = np.zeros((sum(intruder_data_length),1))
    test_label = np.vstack((pos_label,neg_label))

    print("Test data shape: {}, Multi-class Test label shape: {}".format(scaled_test_data.shape,test_label.shape))
    return scaled_test_data,test_label
    