import codecs
import utility_functions_v2 as util
from one_vs_rest_util import get_keystroke_seq
from one_vs_rest_util import get_seq_data
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from collections import Counter

def scale_data(train_data):
    #print("Using standard scaler")
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_data = scaler.transform(train_data)
    return (scaled_data,scaler)

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


def convert_to_keystroke_vec(db):
    timing_vec = util.extract_timing_from_json_testing(db)
    keystroke_vec = util.convert_to_vec(timing_vec)
    return keystroke_vec


def restore_vec(combined_scaled_data,length_list,index):
    start_index = 0
    end_index = 0
    if index!=0:
        for i in range(index):
            start_index = start_index + length_list[i]
            end_index = start_index + length_list[i+1]
        #print("start_index: {}, end_index: {}".format(start_index,end_index))
        keystroke_vec = combined_scaled_data[start_index:end_index]
    else:
        keystroke_vec = combined_scaled_data[0:length_list[0]]
    return keystroke_vec

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
    return (tn, fp, fn, tp), (FAR,FRR,EER)


def siamese_testing(predictions,threshold,y_true):
    predicted_classes = util.classify_predictions(predictions,threshold)
    (tn, fp, fn, tp), (FAR,FRR,EER) = calculate_keystroke_scores(y_true,predicted_classes)
    return predicted_classes,(tn, fp, fn, tp), (FAR,FRR,EER)

def generate_left_right_label_data(user_seq_list,intruder_seq_list):

    #for (user_seq, intruder_seq) in zip(user_seq_list, intruder_seq_list):
    #    user_seq = shuffle(user_seq)
    #    intruder_seq = shuffle(intruder_seq)
    print("not shuffled !!!")
    left_data = np.vstack((user_seq_list[0], user_seq_list[1], \
        user_seq_list[0], user_seq_list[0], user_seq_list[1], user_seq_list[1], \
            user_seq_list[0], user_seq_list[2], user_seq_list[1], user_seq_list[3], \
                user_seq_list[0], user_seq_list[1], user_seq_list[2], user_seq_list[3]))

    right_data = np.vstack((user_seq_list[2], user_seq_list[3], \
        user_seq_list[1], user_seq_list[3], user_seq_list[2], user_seq_list[3], \
            intruder_seq_list[3], intruder_seq_list[0], intruder_seq_list[2], intruder_seq_list[1], \
                intruder_seq_list[0], intruder_seq_list[1], intruder_seq_list[3], intruder_seq_list[2]))

    #make label
    pos_label = np.ones((len(user_seq_list[0])*6,1))
    neg_label = np.zeros((len(user_seq_list[0])*8,1))

    label = np.vstack((pos_label,neg_label))
    
    for i in range(15):
        (left_data,right_data,label) = shuffle(left_data,right_data,label)
    return left_data,right_data,label



def load_siamese_data(user_id, intruder_id, seq_length, num_features = 4):
    #load user keystroke_seq list
    user_seq_list, user_length_list = get_keystroke_seq(user_id, seq_length, True)

    #load intruder keystroke_seq list
    intruder_seq_list, intruder_length_list = get_keystroke_seq(intruder_id, seq_length, True)

    total_length = sum(user_length_list) + sum(intruder_length_list)

    #combine and reshape data to scale
    combined_data = np.vstack((user_seq_list,intruder_seq_list)).reshape(total_length,seq_length*num_features)
    scaled_compiled_data, scaler = scale_data(combined_data)
    scaled_compiled_data = scaled_compiled_data.reshape(total_length,seq_length,num_features)
    scaled_compiled_data = np.expand_dims(scaled_compiled_data, axis=-1)

    scaled_user_data = scaled_compiled_data[0:sum(user_length_list)]
    scaled_intruder_data = scaled_compiled_data[sum(user_length_list):(sum(intruder_length_list)+sum(user_length_list))]

    #restore original shape _user
    user_seq_data = []
    intruder_seq_data = []
    for i in range(4):
        cur_user_seq_data =  restore_vec(scaled_user_data,user_length_list, i)
        curr_intruder_seq_data = restore_vec(scaled_intruder_data,intruder_length_list, i)
        user_seq_data.append(cur_user_seq_data)
        intruder_seq_data.append(curr_intruder_seq_data)
    
    
    user_train_data = []
    user_val_data = []
    intruder_train_data = []
    intruder_val_data = []
    for (user_seq,intruder_seq) in zip(user_seq_data,intruder_seq_data):

        curr_user_train = user_seq[0:1000]
        curr_user_val = user_seq[1000:1400]
        curr_intruder_train = intruder_seq[0:1000]
        curr_intruder_val = intruder_seq[1000:1400]

        user_train_data.append(curr_user_train)
        user_val_data.append(curr_user_val)

        intruder_train_data.append(curr_intruder_train)
        intruder_val_data.append(curr_intruder_val)

    base_user_data_1 = user_train_data[0][0:500]
    base_user_data_2 = user_train_data[2][0:500]

    left_data_train, right_data_train, train_label = generate_left_right_label_data(user_train_data, intruder_train_data)
    left_data_val, right_data_val, val_label = generate_left_right_label_data(user_val_data, intruder_val_data)

    return (left_data_train, right_data_train, train_label), (left_data_val, right_data_val, val_label), scaler, (base_user_data_1,base_user_data_2)

def load_siamese_test_data(user_id, intru_id, scaler, seq_length, base_user_data_1, base_user_data_2, num_features = 4):
    print("comparing first 500 vs first 500 (no random sampled)!!!")
    #load user test data
    db_5 = read_data_for_user(user_id,"2","0","0",make_empty_keystroke_dict())
    db_5_seq_data = get_seq_data(db_5,scaler,seq_length,True)
    db_5_seq_data = np.array(db_5_seq_data).reshape(len(db_5_seq_data), seq_length*num_features)
    db_5_seq_data = scaler.transform(db_5_seq_data)
    db_5_seq_data = db_5_seq_data.reshape(len(db_5_seq_data),seq_length,num_features).tolist()

    db_7 = read_data_for_intruder(intru_id,"2","3","0",make_empty_keystroke_dict())
    db_7_seq_data = get_seq_data(db_7,scaler,seq_length,True)
    db_7_seq_data = np.array(db_7_seq_data).reshape(len(db_7_seq_data), seq_length*num_features)
    db_7_seq_data = scaler.transform(db_7_seq_data)
    db_7_seq_data = db_7_seq_data.reshape(len(db_7_seq_data),seq_length,num_features).tolist()

    from random import sample
    #pos_data = sample(db_5_seq_data,500)
    #neg_data = sample(db_7_seq_data,500)
    pos_data = db_5_seq_data[0:500]
    neg_data = db_7_seq_data[0:500]

    right_data_test = np.vstack((pos_data,neg_data,pos_data,neg_data))
    right_data_test = np.expand_dims(right_data_test, axis=-1)

    left_data_test = np.vstack((base_user_data_1,base_user_data_1,base_user_data_2,base_user_data_2))

    pos_label = np.ones((500,1))
    neg_label = np.zeros((500,1))
    test_label = np.vstack((pos_label,neg_label,pos_label,neg_label))

    return (left_data_test,right_data_test,test_label), (pos_data,neg_data)

def load_train_data(user_id,intruder_id,seq_length):
    #load user data
    user_db_1 = read_data_for_user(user_id,"2","0","1",make_empty_keystroke_dict())
    user_db_2 = read_data_for_user(user_id,"0","0","1",make_empty_keystroke_dict())
    user_db_3 = read_data_for_user(user_id,"1","0","0",make_empty_keystroke_dict())
    user_db_4 = read_data_for_user(user_id,"1","0","1",make_empty_keystroke_dict())

    user_db_1_keystroke_vec = convert_to_keystroke_vec(user_db_1)
    user_db_2_keystroke_vec = convert_to_keystroke_vec(user_db_2)
    user_db_3_keystroke_vec = convert_to_keystroke_vec(user_db_3)
    user_db_4_keystroke_vec = convert_to_keystroke_vec(user_db_4)

    user_length_list = [len(user_db_1_keystroke_vec),len(user_db_2_keystroke_vec),len(user_db_3_keystroke_vec),len(user_db_4_keystroke_vec)]
    temp_user_list = user_db_1_keystroke_vec + user_db_2_keystroke_vec + user_db_3_keystroke_vec + user_db_4_keystroke_vec 

    #load intruder data
    intruder_db_1 = read_data_for_intruder(intruder_id,"2","3","1",make_empty_keystroke_dict())
    intruder_db_2 = read_data_for_intruder(intruder_id,"0","1","1",make_empty_keystroke_dict())
    intruder_db_3 = read_data_for_intruder(intruder_id,"1","2","0",make_empty_keystroke_dict())
    intruder_db_4 = read_data_for_intruder(intruder_id,"0","1","0",make_empty_keystroke_dict())

    intruder_db_1_keystroke_vec = convert_to_keystroke_vec(intruder_db_1)
    intruder_db_2_keystroke_vec = convert_to_keystroke_vec(intruder_db_2)
    intruder_db_3_keystroke_vec = convert_to_keystroke_vec(intruder_db_3)
    intruder_db_4_keystroke_vec = convert_to_keystroke_vec(intruder_db_4)

    intruder_length_list = [len(intruder_db_1_keystroke_vec),len(intruder_db_2_keystroke_vec),len(intruder_db_3_keystroke_vec),len(intruder_db_4_keystroke_vec)]
    temp_intruder_list = intruder_db_1_keystroke_vec + intruder_db_2_keystroke_vec + intruder_db_3_keystroke_vec + intruder_db_4_keystroke_vec 

    #combine and scale data
    compiled_data = temp_user_list + temp_intruder_list
    scaled_compiled_data, scaler = scale_data(compiled_data)
    scaled_user_data = scaled_compiled_data[0:sum(user_length_list)]
    scaled_intruder_data = scaled_compiled_data[sum(user_length_list):(sum(intruder_length_list)+sum(user_length_list))]

    #restore original shape _user
    user_db_1_keystroke_vec = restore_vec(scaled_user_data,user_length_list,0)
    user_db_2_keystroke_vec = restore_vec(scaled_user_data,user_length_list,1)
    user_db_3_keystroke_vec = restore_vec(scaled_user_data,user_length_list,2)
    user_db_4_keystroke_vec = restore_vec(scaled_user_data,user_length_list,3)
    
    #convert to user_seq
    user_db_1_seq_data = util.vec_to_seq(user_db_1_keystroke_vec,seq_length)
    user_db_2_seq_data = util.vec_to_seq(user_db_2_keystroke_vec,seq_length)
    user_db_3_seq_data = util.vec_to_seq(user_db_3_keystroke_vec,seq_length)
    user_db_4_seq_data = util.vec_to_seq(user_db_4_keystroke_vec,seq_length)
    
    user_data = user_db_1_seq_data + user_db_2_seq_data + user_db_3_seq_data + user_db_4_seq_data
    
    #restore original shape _intruder
    intruder_db_1_keystroke_vec = restore_vec(scaled_intruder_data,intruder_length_list,0)
    intruder_db_2_keystroke_vec = restore_vec(scaled_intruder_data,intruder_length_list,1)
    intruder_db_3_keystroke_vec = restore_vec(scaled_intruder_data,intruder_length_list,2)
    intruder_db_4_keystroke_vec = restore_vec(scaled_intruder_data,intruder_length_list,3)
    
    #convert to intruder_seq
    intruder_db_1_seq_data = util.vec_to_seq(intruder_db_1_keystroke_vec,seq_length)
    intruder_db_2_seq_data = util.vec_to_seq(intruder_db_2_keystroke_vec,seq_length)
    intruder_db_3_seq_data = util.vec_to_seq(intruder_db_3_keystroke_vec,seq_length)
    intruder_db_4_seq_data = util.vec_to_seq(intruder_db_4_keystroke_vec,seq_length)
    
    intruder_data = intruder_db_1_seq_data + intruder_db_2_seq_data + intruder_db_3_seq_data + intruder_db_4_seq_data
    
    train_data,train_labels = prep_data(user_data,intruder_data,4)
    print("train_data shape: {}, train_labels shape: {}".format(train_data.shape,train_labels.shape))
    return train_data,train_labels

def load_test_data(user_id,intru_id,scaler,seq_length):
    #load user test data
    db_5 = read_data_for_user(user_id,"2","0","0",make_empty_keystroke_dict())
    db_5_seq_data = get_seq_data(db_5,scaler,seq_length)


    db_7 = read_data_for_intruder(intru_id,"2","3","0",make_empty_keystroke_dict())
    db_7_seq_data = get_seq_data(db_7,scaler,seq_length)
    
    from random import sample
    pos_data = sample(db_5_seq_data,500)
    neg_data = sample(db_7_seq_data,500)
    
    test_data,test_label = prep_data(pos_data,neg_data,4)
    print("test_data shape: {}, test_label shape: {}".format(test_data.shape,test_label.shape))
    return test_data,test_label