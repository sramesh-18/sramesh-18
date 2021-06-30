#!/usr/bin/env python
# coding: utf-8

# ## We're going to be (re) training a neural network to identify specific genes in NGS sequencing in Tensorflow 2! WOW

training_params = {
    'num_samples': 1000, # number of samples to take frome each round of sampling
    'batch_size': 100, 
    'nb_epochs' : 2000,
    'learning_rate': 1e-4,
    'n_prob':.06, # 0.06 the chance that nucleotide will be replaced with 'N'
    'mutation_prob':.02, # 0.02 the chance that any given nucleotide will be replaced with another random nucleotide
    'delete_prob':.015, # 0.015 the chance that any given nucleotide will be deleted
    'insert_prob':.015,# 0.015 the chance that any given space between 2 nucleotides will have one inserted
    'min_read_length':75, # minimum read length
    'max_read_length':150, # 250 max read length
    'sample_size': 5 #num lines to read from
}

output_classes = {
    #'hla_a'  :1,
    #'hla_b'  :1,
    #'hla_c'  :2,
    #'hla_dp' :2,
    #'hla_dq' :3,
    #'hla_dr' :4,
    #'hla_tap':6,
    #'hla_mica':7,
    #'hla_micb':8,
    #'hla_kir':9
    #'refMrna' :10
    #'hla': 1,
    #'non-hla': 0
}

output_paths= {
    #'hla_a'  :"/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/hla_a2.fastm",
    #'hla_b'  :"/Users/davidott/Workspace/notebooks/test/hla_fastm/hla_b.fastm",
    #'hla_c'  :"/Users/davidott/Workspace/notebooks/test/hla_fastm/hla_c.fastm",
    #'hla_dp' :"/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/hla_dp2.fastm",
    #'hla_dq' :"/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/hla_dq2.fastm",
    #'hla_dr' :"/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/hla_dr2.fastm",
    #'hla_tap':"/Users/davidott/Workspace/notebooks/test/hla_fastm/hla_tap.fastm",
    #'hla_mica':"/Users/davidott/Workspace/notebooks/test/hla_fastm/hla_mica.fastm",
    #'hla_micb':"/Users/davidott/Workspace/notebooks/test/hla_fastm/hla_micb.fastm",
    #'hla_kir':"/root/Notebooks/test/hla_fastm/KIR_nuc.fastm"
    #'hla': "/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/hla_dr2.fastm_copy",
    #'non-hla': "/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/hla_dr2.fastm_copy"
    #'hla':"/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/hla.fastm",
    #'non-hla': "/Users/shreyaramesh/Downloads/Shreya_Data/hla_fastm_copy/refMrna.fastm"
    
}


# ### Here are a few functions to implement:

# In[3]:


#Import data and other packages
#Import data and other packages
import pandas as pd
import numpy as np
import random

#Handling probabilities
    
def random_mutation(read, probability):
    length = len(read)
    my_arr = np.random.choice([0, 1], size=(length,), p=[1 - probability, probability])
    return "".join([np.take(["A", "C", "G", "T"], random.randint(0, 3)) if my_arr[i] == 1 else read[i] for i in range(length)])

def random_n_mutation(read, probability):
    length = len(read)
    my_arr = np.random.choice([0, 1], size=(length,), p=[1 - probability, probability])
    return "".join([np.take(["N"], random.randint(0, 0)) if my_arr[i] == 1 else read[i] for i in range(length)])

def random_delete(read, probability):
    length = len(read)
    my_arr = np.random.choice([0, 1], size=(length,), p=[1 - probability, probability])
    return "".join([np.take([""], random.randint(0, 0)) if my_arr[i] == 1 else read[i] for i in range(length)])
    
def random_insert(read, probability):
    length = len(read)
    my_arr = np.random.choice([0, 1], size=(length,), p=[1 - probability, probability])
    return "".join([np.take(["A"+ read[i], "C"+ read[i], "G"+ read[i], "T" + read[i]], random.randint(0, 3)) if my_arr[i] == 1 else read[i] for i in range(length)])

#Splitting the transcripts into smaller datasets 

def sample_from_transcripts(file, training_params, training_output):
    completed_array = []
    completed_array_class =[]
    dataframe = pd.read_csv(file, header = None, delimiter = 'bp,')
    converted_df = dataframe[1]
    
    #creating a list of lists
    
    bp_length = training_params.get('max_read_length')
    for row in converted_df:
        i = 0
        if len(row) < bp_length:
            row = row.ljust(bp_length, 'N')
    
        while len(row) - i >= bp_length:
            
            new_string = row[i:bp_length+i]
            
            new_string = random_mutation(new_string, training_params['mutation_prob'])
            new_string = random_n_mutation(new_string, training_params['n_prob'])
            new_string = random_delete(new_string, training_params['delete_prob'])
            new_string = random_insert(new_string, training_params['insert_prob'])
            completed_array.append(new_string)
            #using the training_output to append to completed_array_class 
            completed_array_class.append(training_output)
            
            i = i + 1

     
    max_transcript_length = len(max(completed_array, key = len))
    completed_array = [entry.ljust(max_transcript_length, "N") for entry in completed_array]
    return completed_array,completed_array_class

#file = 'data/Sample/hla_a2.fastm'

#transcript_array = sample_from_transcripts(file, training_params, 0)
#print(transcript_array)

#Converting data into numpy array
def reads_to_numpy(input):#reads, training_parameters):
    final_array = []
    i = 0
    for index in input:
        string_array = []
        #print(index)
        #[final_array.append(string_array) for index in input]
        for i in index:
            #print(i)
            if i == "A":
                string_array.append([1,0,0,0])
            elif i == "C":
                string_array.append([0,1,0,0])
            elif i == "G":
                string_array.append([0,0,1,0])
            elif i == "T":
                string_array.append([0,0,0,1])
            else:
                string_array.append([0,0,0,0])
        final_array.append(string_array)
        tensor_array= np.array(final_array)
    return tensor_array    

#tensor_array = reads_to_numpy(transcript_array)

#print(tensor_array)



total_tensor_class = []
total_tensor_array = []

for index in output_paths:
    temp_array = []
    temp_combo = sample_from_transcripts(output_paths[index], training_params, output_classes[index])
    temp_array = temp_combo[0]
    temp_class = temp_combo[1]
    
    total_tensor_class = total_tensor_class+temp_class
    total_tensor_array = total_tensor_array+temp_array


#total_tensor_array = reads_to_numpy(total_tensor_array)
#total_tensor_class = np.asarray(total_tensor_class).astype('float32')

#print(total_tensor_array)
#print(total_tensor_class)

### so our results will look like this:
'''
training_params['num_samples'] = 3
training_params['max_read_length'] = 10
reads = ['ACGT',
        'ACCCTGNT',
        'NNAGTCTCAT'
        ]
reads_to_numpy(reads)
"""
array([[[1,0,0,0],  # 'ACGT'
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]],
        
       [[1,0,0,0], # 'ACCCTGNT'
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,0,0,0],
        [0,0,0,1],
        [0,0,0,0],
        [0,0,0,0]],
        
       [[0,0,0,0],  #  'NNAGTCTCAT'
        [0,0,0,0],
        [1,0,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,1,0,0],
        [0,0,0,1],
        [0,1,0,0],
        [1,0,0,0],
        [0,0,0,1]]]
)

"""
'''


# In[ ]:





# In[ ]:




