# covseq2seq
Seq2Seq model for simulating covid genome lineage evolution

This is an archive repository for self-learning purposes, documenting my work and first time experience with machine learning models. I aim to clean it up and provide a working example.


The idea and workflow:

1. create_dataset.py Creates the training dataset from when you have multiple weeks and folders organized as
2020/Jan2020/KWs etc ---> this creates the base data_input.pkl and data_target.pkl files, that look like

['C241T T445C C1513T C3037T C6286T C14408T G21255C C22377T A23403G C25000T C26801G C28932T G29645T', ...]

TO DO: save their lineage alongside - as dictionary (not the same as the pairs!!) 

1.2 get_test_data.py
create a test set using only one dataframe without pairs ---> creates directly the input tensor file  

2. get_train_data.py
if the create_dataset has been used, use this to process the pairs into input and target tensors
current dimensions: (, 94) (, 36)

3.1 train_model_gpu.py
workflow to train the model

3.2 simulator.py
simulate the entwicklung  
