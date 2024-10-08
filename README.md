# covseq2seq
Seq2Seq model for simulating covid genome lineage evolution

This is an archive repository for self-learning purposes, documenting my work and first time experience with machine learning models. I aim to clean it up and provide a working example. It is a sequence to sequence model, analysing the input mutations and trying to predict what new mutations can emerge.


The idea and workflow:

1. create_dataset.py \
Create the training dataset. The folders are organized as 2020/Jan2020/KWs. The script loops through the individual KW (calender weeks) and gathers the mutations ---> this creates the base data_input.pkl and data_target.pkl files, that look like: \
['C241T T445C C1513T C3037T C6286T C14408T G21255C C22377T A23403G C25000T C26801G C28932T G29645T', ...] \
where one string is a collection of mutations separated by a blank space.
2. get_test_data.py \
Create a test set using only one dataframe ---> creates directly the input tensor file.
3. get_train_data.py \
If create_dataset has been used, use this to process the input and target pairs into input and target tensors. Current dimensions: (, 94) (, 36)
4. train_model_gpu.py \
Workflow to train the model, adjusted to make use of a gpu.
5. simulator.py \
Simulate the development, feeding the results back as inputs until a certain threshold has been met. 
