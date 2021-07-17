#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sfseventools.training import FastmSampler, sample_from_transcripts, build_train_sets, shuffle_x_y_together
import numpy as np
import sys
import random

# To see whole NP array. USE WITH CAUTION
# np.set_printoptions(threshold=sys.maxsize)


# In[15]:


uid = ">"
sep =","
ender = ";"
fastm_samplers = []

file_paths ={
#ref_rna_path = {
    #'hla_a'  :np.asarray([1,0,0,0,0,0,0,0,0]),
    #'hla_b'  :np.asarray([0,1,0,0,0,0,0,0,0]),
    #'hla_c'  :np.asarray([0,0,1,0,0,0,0,0,0]),
    #'hla_dp' :np.asarray([0,0,0,1,0,0,0,0,0]),
    #'hla_dq' :np.asarray([0,0,0,0,1,0,0,0,0]),
    #'hla_dr' :np.asarray([0,0,0,0,0,1,0,0,0]),
    #'hla_tap':np.asarray([0,0,0,0,0,0,1,0,0]),
    #'hla_mica':np.asarray([0,0,0,0,0,0,0,1,0]),
    #'hla_micb':np.asarray([0,0,0,0,0,0,0,0,1]),
    #'hla_kir':np.asarray([0,0,0,0,0,0,0,0,1])
    #*'hla_a':np.asarray([1,0,0,0,0]),
    #*'hla_dp':np.asarray([0,1,0,0,0]),
    #*'hla_dq':np.asarray([0,0,1,0,0]),
    #*'hla_dr':([0,0,0,1,0]),
    #'hla_dr_copy':np.asarray([0,0,0,0,1])
    #*'non-hla': ([0,0,0,0,1])
    #'neither': ([0,0,0,0,0,0])
    #'sample': ([0,0,0,0,1])
    'hla_a'  :1,
    #'hla_b'  :1,
    #'hla_c'  :2,
    'hla_dp' :2,
    'hla_dq' :3,
    'hla_dr' :4,
    #'hla_tap':6,
    #'hla_mica':7,
    #'hla_micb':8,
    #'hla_kir':9
    'non-hla': 0
    
    
}
output_paths= {
    'hla_a'  :"data/Sample/hla_a2.fastm",
    #'hla_b'  :"data/Sample/hla_b2.fastm",
    #'hla_c'  :"data/Sample/hla_c2.fastm",
    'hla_dp' :"data/Sample/hla_dp2.fastm",
    'hla_dq' :"data/Sample/hla_dq2.fastm",
    'hla_dr' :"data/Sample/hla_dr2.fastm",
    #'hla_tap':"data/Sample/hla_tap2.fastm",
    #'hla_mica':"data/Sample/hla_mica2.fastm",
    #'hla_micb':"data/Sample/hla_micb2.fastm",
    #'sample'  :"data/Original/sample.fastm"
    'non-hla' :"data/Sample/refMrna.fastm"
}


with open("data/Original/sample.fastm") as myfile:
    head = [next(myfile) for x in range(2)]
#print(head)


# In[16]:


training_params = {
    'num_samples': 1000,
    'batch_size': 100,
    'nb_epochs' : 2000,
    'learning_rate': 1e-4,
    'n_prob_upper_limit':.06,
    'mutation_prob_upper_limit':.02,
    'delete_prob_upper_limit':.015,
    'insert_prob_upper_limit':.015,
    'min_read_length':75,
    'b_randomize_location':True,
    'b_randomize_direction':False
}

output_classes = {
    'hla_a'  :1,
    #'hla_b'  :1,
    #'hla_c'  :2,
    'hla_dp' :2,
    'hla_dq' :3,
    'hla_dr' :4,
    #'hla_tap':6,
    #'hla_mica':7,
    #'hla_micb':8,
    #'hla_kir':9
    'non-hla': 0
    #'sample':0
}



mp = {
        'max_read_length':400,
        'loss':'binary_crossentropy',
        'optimizer':'rmsprop',
        'metrics':['accuracy'],
        'num_classes': len(output_classes.keys())
    }
    
'''
training_params = {
    'num_samples': 1000,
    'batch_size': 100,
    'nb_epochs' : 2000,
    'learning_rate': 1e-4,
    'n_prob_upper_limit':.06,
    'mutation_prob_upper_limit':.02,
    'delete_prob_upper_limit':.015,
    'insert_prob_upper_limit':.015,
    'min_read_length':75,
    'b_randomize_location':False,
    'b_randomize_direction':False
}
'''

output_number = len(output_classes)
print(output_number)


# In[18]:


num_classes = len(output_classes.keys())
print(num_classes)
for train_class, idx in output_classes.items():
    output_arr = np.zeros(num_classes)
    output_arr[idx] = 1
    train_sampler = FastmSampler(fastm_file_path=output_paths[train_class],
                        n_prob_upper_limit=training_params['n_prob_upper_limit'],
                        mutation_prob_upper_limit=training_params['mutation_prob_upper_limit'],
                        delete_prob_upper_limit=training_params['delete_prob_upper_limit'],
                        insert_prob_upper_limit=training_params['insert_prob_upper_limit'],
                        min_read_length=training_params['min_read_length'],
                        tensor_length=mp['max_read_length'],
                        b_randomize_location=training_params['b_randomize_location'],
                        b_randomize_direction=training_params['b_randomize_direction'],
                        training_output=output_arr,
                        uid=uid,
                        sep=sep,
                        ender = ender
                       )
    print(train_class,idx,output_arr)
    fastm_samplers.append(train_sampler)


# In[19]:


X_train,y_train,X_test,y_test = build_train_sets(data_samplers=fastm_samplers,
                  sample_size = training_params['num_samples'])


# In[20]:


X_train, y_train = shuffle_x_y_together(X_train,y_train)


# In[21]:


random.shuffle(X_train)
random.shuffle(X_test)
random.shuffle(y_train)
random.shuffle(y_test)

#x_data = np.concatenate((X_train,X_test),axis=0)
#y_data = np.concatenate((y_train,y_test),axis=0)

# In[25]:


#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)


# In[ ]:




