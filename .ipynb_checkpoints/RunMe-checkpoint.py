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
    'hla_a'  :0,
    'hla_b'  :1,
    'hla_c'  :2,
    'hla_dp' :3,
    'hla_dq' :4,
    'hla_dr' :5,
    'hla_tap':6,
    'hla_mica':7,
    'hla_micb':8,
    'non-hla': 9
    
}
output_paths= {
    'hla_a'  :"/home/jovyan/data/Original/hla_a.fastm",
    'hla_b'  :"/home/jovyan/data/Original/hla_b.fastm",
    'hla_c'  :"/home/jovyan/data/Original/hla_c.fastm",
    'hla_dp' :"/home/jovyan/data/Original/hla_dp.fastm",
    'hla_dq' :"/home/jovyan/data/Original/hla_dq.fastm",
    'hla_dr' :"/home/jovyan/data/Original/hla_dr.fastm",
    'hla_tap':"/home/jovyan/data/Original/hla_tap.fastm",
    'hla_mica':"/home/jovyan/data/Original/hla_mica.fastm",
    'hla_micb':"/home/jovyan/data/Original/hla_micb.fastm",
    'non-hla':"/home/jovyan/data/Sample/refMrna.fastm"
}


with open("data/Original/sample.fastm") as myfile:
    head = [next(myfile) for x in range(2)]


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
    'hla_a'  :0,
    'hla_b'  :1,
    'hla_c'  :2,
    'hla_dp' :3,
    'hla_dq' :4,
    'hla_dr' :5,
    'hla_tap':6,
    'hla_mica':7,
    'hla_micb':8,
    'non-hla': 9
}

mp = {
        'max_read_length':150,
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
                        training_output=idx,
                        uid=uid,
                        sep=sep,
                        ender = ender
                       )
    fastm_samplers.append(train_sampler)


# In[19]:


X_train,y_train,X_test,y_test = build_train_sets(data_samplers=fastm_samplers,
                  sample_size = training_params['num_samples'])


# In[20]:


#X_train, y_train = shuffle_x_y_together(X_train,y_train)


# In[21]:

x_data = np.concatenate((X_train,X_test),axis=0)
y_data = np.concatenate((y_train,y_test),axis=0)


# In[25]:

# In[ ]:




