DATASET CAN BE FOUND HERE: http://times.cs.uiuc.edu/~wang296/Data/
PLEASE MAKE CITATION IF YOU ARE USING THE DATA!!

1. deep_learning contains all self-implemented deep CNN models. 
- model.py 
define function of deep parallel convolutional neural network 
- CNN_word2vec_run.py
Run to start training deep parallel convolutional neural network (result reported)
- CNN_word2vec_2class.py
Run to start training deep nested convolutional neural network for 2 classes classification(implemented but not tested)
- CNN_word2vec_5class.py
Run to start training deep nested convolutional neural network for 5 classes classification(implemented but not tested)

2.machine_learning contains all machine learning model train and test results.
- ml_models.py
Run to get result tables in the report

3. word_embedding contains all feature extraction methods 
- lda.py
Run to get lda document distributions as feature matrix
- pre_train_word2vec.py
Use google news pretained word2vec embedding. Need to have google news pertained word2vec bin in folder. Run to get word2vec results

4.helper_file contains all helper python files needed to generate required input 
- save_required_input.py
Run to generated required csv files
- generate_random_index.py
Run to generated random index file needed for the program
- Ensemble_based_split.py
Run to split reviews and generated related csv files
- generate_wordcloud.R
R code for generating word clouds. Run to generate word clouds for csv files. 
