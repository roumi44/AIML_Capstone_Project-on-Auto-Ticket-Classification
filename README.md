# AIML_Capstone_Project-on-Auto-Ticket-Classification
1 Objective:
The objective of the project is to build a classifier that can classify the tickets to right group by analyzing the text. We will concentrate on the below objectives also for building a good classifier.
•	EDA and visualizations are to be done for understanding the importance of each column and to have proper conclusion on the data set.
•	The data has to be checked for inconsistencies if any then tackle those in an efficient way.
•	Different natural language processing, preprocessing techniques are tried using maximum of our features.
•	Different feature extraction techniques and comparison of models.
•	How to use different classification models have to be explored.
•	How to use transfer learning to use pre-built models have to be explored.
•	Learn to set the optimizers, loss functions, epochs, learning rate, batch size, checkpointing, early stopping etc.
• Model Deployment


2.Solution:
2.1.Common Library and functions and installation:
2.1.1.Mount Drive

2.1.2.Installation

2.1.3.Import all libraries

2.1.4.Write all common functions

2.2.Explore the input data provided through excel:
2.2.1.EDA included shape,description,info,Print Top 5 rows

2.3.Missing points in data:
2.3.1.Find all the missing values and display them.

2.4.Visualizing different patterns:
2.4.1.Assignment group target column exploration and visualization with top 10 groups,Unique groups,Total group counts,Word cloud visualizations for assignment group.

2.4.2.Caller column exploration with plot for multiple groups for top 10 caller

2.4.3.Max word and min word and plot of word distribution for description and short description column

2.5.Dealing with missing values:
2.5.1.We will create a combine column and there we will conatinate two groups as NAN + string is NAN so we will use THE at place of NAN values which will be removed later with stop word removal

2.6.Finding inconsistencies in the data:
2.6.1.We will try lang detect and polyglot library and will see language accuracy.

2.6.2.We will divide the dataset to two part one with having rows with english language and one with non english languages.

2.6.3.Then we will apply Google translation on non english rows and merge the non english part with english part and create final feature column

2.6.4.We will check language accuracy after translation


2.7.Visualizing different text features:
2.7.1.And visualize final feature column in word cloud with and without stop words

2.7.2.And also calculate top 50 frequent word

2.7.3.Observations

2.8.Text preprocessing:
2.8.1.We will do preprocessing and removed digit special character url mailID Extra spaces accented chars newlines HTML and also expand contractions and lemmatization will also be applied.We have created a single function inside it.

2.8.2.Observations

2.9.Creating tokens as required
2.9.1We will tokenize the final feature and remove stop word and have final tokenized feature for embedding.

2.10.Pickle data set
3.Model Building ML MODEL
3.1.Featurization
3.1.1. Convert the target column to label encoded output

3.1.2.Try TFIDF and Count vectorizaer and apply transformed input to different conventional Machine Learning models with test train split

3.1.3. Creation of Data frame to compare the models

3.1.4.Observations

3.2.Building a model architecture which can classify and trying different model architectures by researching state of the art for similar tasks for count vectorizer
3.2.1.Try multiple models for transformed input with Count vectorizer and compare their performence.Plot Accuracy,Precision F1 score and Recall comparison plot for different models.

3.2.2.Creation of a common function on model building with all the performence metrices

3.2.3.Sample classification report of ML models

3.2.4.Observations

3.3.Building a model architecture which can classify and trying different model architectures by researching state of the art for similar tasks for tfidf vectorizer
3.3.1.Try multiple models for transformed input with tfidf vectorizer and compare their performence.Plot Accuracy,Precision F1 score and Recall comparison plot for different models.

3.3.2.Comparison data frame of different machine learning models

3.3.3.Observations

3.4.Building a model architecture which can classify and trying different model architectures by researching state of the art for similar tasks for Pipeline with Count then TIDF vectorizer
3.4.1.Try multiple models for transformed input with Pipeline with Count then TIDF vectorizer and compare their performence.Plot Accuracy,Precision F1 score and Recall comparison plot for different models.

3.4.2.Comparison data frame of different machine learning models

3.4.3.Observations

Complete data set
3.5.Featurization
3.5.1. Convert the target column to label encoded output

3.5.2.Try TFIDF and Count vectorizaer and apply transformed input to different conventional Machine Learning models with test train split

3.5.3. Creation of Data frame to compare the models

3.5.4.Observations

3.6.Building a model architecture which can classify and trying different model architectures by researching state of the art for similar tasks for count vectorizer
3.6.1.Try multiple models for transformed input with Count vectorizer and compare their performence.Plot Accuracy,Precision F1 score and Recall comparison plot for different models.

3.6.2.Creation of a common function on model building with all the performence metrices

3.6.3.Sample classification report of ML models

3.6.4.Observations

3.7.Building a model architecture which can classify and trying different model architectures by researching state of the art for similar tasks for tfidf vectorizer
3.7.1.Try multiple models for transformed input with tfidf vectorizer and compare their performence.Plot Accuracy,Precision F1 score and Recall comparison plot for different models.

3.7.2.Comparison data frame of different machine learning models

3.7.3.Observations

3.8.Building a model architecture which can classify and trying different model architectures by researching state of the art for similar tasks for Pipeline with Count then TIDF vectorizer
3.8.1.Try multiple models for transformed input with Pipeline with Count then TIDF vectorizer and compare their performence.Plot Accuracy,Precision F1 score and Recall comparison plot for different models.

3.8.2.Comparison data frame of different machine learning models

3.8.3.Observations

4.Model building Deep learning model
Top 5 Data
Glove 100D
4.1.Create the deep learning base model.
4.1.1.Building of basic model with one Embedding,LSTM and Dense layer without weight matrix to check the accuracy.

4.1.2.Tokenization and vocab creation

4.1.3.Base Model with categorical_crossentropy and adam optimizer

4.1.4. Prediction

4.1.5.Error Plot

4.1.6 Prediction block and classification report

4.1.7.Observation

4.2.Create the same deep learning base model with drop out
4.3.Create the deep learning base model with glove 100d weight matrix
4.3.1.Building of basic model with one Embedding,LSTM and Dense layer with weight matrix to check the accuracy.

4.3.2.Tokenization and vocab creation

4.3.3.Base Model with categorical_crossentropy and adam optimizer

4.3.4.Error Plot

4.3.5 Prediction block and classification report

4.3.6.Observation

4.4.Create one layer bidirectional LSTM deep learning model with glove 100
4.4.1. Building of a model with one Embedding with glove 100d weight matrix, one bidirectional LSTM and one Dense layer to check the accuracy.

4.4.2.Bi directional LSTM Model with categorical_crossentropy and adam optimizer

4.4.3.Error Plot

4.4.4. Prediction block and classification report

4.4.5.Observation

4.5.Create one layer bidirectional LSTM deep learning model with flatten and return sequence as True and glove 100 embedding
4.5.1. Building of a model with one Embedding layer with glove 100d weight matrix, one bidirectional LSTM with return sequence as True and one flatten and one Dense layer.

4.5.2.Model Description

4.5.3.Error Plot

4.5.4. Prediction block and classification report

4.5.5.Observation

4.6.Create one layer bidirectional LSTM deep learning model with GlobalMaxPool1D and return sequence as True and glove 100 embedding and 3 dense layers
4.6.1. Building of a model with one Embedding layer with glove 100d weight matrix, one bidirectional LSTM with return sequence as True,1 globalmaxpool1D,3 Dense layers.

4.6.2.Model Description

4.6.3.Error Plot

4.6.4. Prediction block and classification report

4.6.5.Observation

4.7.Create two layers bidirectional LSTM deep learning model with return sequence as True and glove 100 embedding and 1 dense layer and 1 flatten layer
4.7.1. Building of a model with one Embedding layer with glove 100d weight matrix, two bidirectional LSTM layers with return sequence as True,1 flatten and 1 Dense layer.

4.7.2.Model Description

4.7.3.Error Plot

4.7.4. Prediction block and classification report

4.7.5.Observation

4.8.Create two layers bidirectional LSTM deep learning model with return sequence as True and glove 100 embedding and 1 dense layer and 1 flatten layer ,1 drop out layer and 1 spatial drop out layer
4..8.1. Building of a model with one Embedding layer with glove 100d weight matrix, two bidirectional LSTM layers with return sequence as True,1 flatten and 1 Dense layer, 1 drop out layer and 1 spatial drop out layer

4.8.2.Model Description

4.8.3. Prediction block and classification report

4.8.5.Error Plot

5.1.Create the deep learning base model.
5.1.1.Building of basic model with one Embedding,LSTM and Dense layer without weight matrix to check the accuracy.

5.1.2.Tokenization and vocab creation

5.1.3.Base Model with categorical_crossentropy and adam optimizer

5.1.4. Prediction

5.1.5.Error Plot

5.1.6 Prediction block and classification report

5.1.7.Observation

5.2.Create the deep learning base model with glove 100d weight matrix
5.2.1.Building of basic model with one Embedding,LSTM and Dense layer with weight matrix to check the accuracy.

5.2.2.Tokenization and vocab creation

5.2.3.Base Model with categorical_crossentropy and adam optimizer

5.2.4.Error Plot

5.2.5 Prediction block and classification report

5.2.6.Observation

5.3.Create one layer bidirectional LSTM deep learning model with glove 100
5.3.1. Building of a model with one Embedding with glove 100d weight matrix, one bidirectional LSTM and one Dense layer to check the accuracy.

5.3.2.Bi directional LSTM Model with categorical_crossentropy and adam optimizer

5.3.3.Error Plot

5.3.4. Classiication report

5.3.5.Observation

5.4.Create one layer bidirectional LSTM deep learning model with flatten and return sequence as True and glove 100 embedding
5.4.1. Building of a model with one Embedding layer with glove 100d weight matrix, one bidirectional LSTM with return sequence as True and one flatten and one Dense layer.

5.4.2.Model Description

5.4.3.Error Plot

5.4.4. Classification report

5.4.5.Observation

5.5.Create one layer bidirectional LSTM deep learning model with GlobalMaxPool1D and return sequence as True and glove 100 embedding and 3 dense layers
5.5.1. Building of a model with one Embedding layer with glove 100d weight matrix, one bidirectional LSTM with return sequence as True,1 globalmaxpool1D,3 Dense layers.

5.5.2.Model Description

5.5.3.Error Plot

5.5.4. Classification report

5.5.5.Observation

5.6.Create two layers bidirectional LSTM deep learning model with return sequence as True and glove 100 embedding and 1 dense layer and 1 flatten layer
5.6.1. Building of a model with one Embedding layer with glove 100d weight matrix, two bidirectional LSTM layers with return sequence as True,1 flatten and 1 Dense layer.

5.6.2.Model Description

5.6.3.Error Plot

5.6.4. Classification report

5.6.5.Observation

5.7.Create two layers bidirectional LSTM deep learning model with return sequence as True and glove 100 embedding and 1 dense layer and 1 flatten layer ,1 drop out layer and 1 spatial drop out layer
5.7.1. Building of a model with one Embedding layer with glove 100d weight matrix, two bidirectional LSTM layers with return sequence as True,1 flatten and 1 Dense layer, 1 drop out layer and 1 spatial drop out layer

5.7.2.Model Description

5.7.3.Error Plot

5.7.4. Classification report

5.7.5.Observation

6.Final model check for Deployment

Pretrained Model
7.Fasttext
8.ULM FIT

9.Conclusion
For this problem statement we first completed text preprocessing and perform all the activities explained in the notebook with complete data set and what we observed that the results and predictions are not good as there is a hudge data imbalance if we see classification report of complete data set for the top models we can see for some groups the F1 csore came as 0 and there is a hudge imbalance and also 5000+ data with top 5 group where as we have 8500 total data and 74 total groups which cleary indicates data is notbalanced.

Next we ran same steps for top 5 models and we saw hudge change in model scores Train reaches to 96 where as test too reaches 93 which is really good and proves the imbalance issue.

We have selected model 009 , ULM fit and Fast text and BERT for top 5 models as the performence are really good.

Further we have deployed the model009 through flask to webpage
