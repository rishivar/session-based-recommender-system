
# A Session Based Recommender System using LSTM Network.

Contributors: [Srinivas Thillaisthanam Raman](https://www.linkedin.com/in/srinivas-thillaisthanam-raman-6908b0a2/), [Rishivardhan Krishnamoorthy](https://www.linkedin.com/in/rishi-vardhan/) 

The repository contains the code for a Session based recommender system using a Long Short Term Memory Network which was built as a part of the course project for CSE258 Recommender Systems at UC San Diego.
Session-based recommender systems are used to predict the next item that a user would purchase given the set of items that the user has clicked/viewed
within a session. The model is trained and evaluated on the Recsys 2022 challenge dataset. (https://recsys.acm.org/recsys22/).

The dataset is sorted based on the timestamp. Sessions from January 2020 to April 2021 are used for training and sessions that belong to May 2021 are used for testing. 
The training set contains 918382 sessions and the test set contains 81618 sessions. 
Mean Reciprocal Rank is used as the evaluation metric for this problem. This was the metric used to evaluate the performances of submissions in the Recsys 2022 challenge. 

For every test sample, the top 100 recommendations are fetched from the model and sorted based on their probabilities. 
If the ground truth is present in the recommendations returned by the model, the reciprocal rank is calculated for the sample. Otherwise, it is considered as zero.


**Baseline Model**

A Word2vec Continuous Bag Of Words (CBOW) model is used as the baseline model for this problem. We train a CBOW model to predict an item in a session given the other items that were viewed or purchased in the session This lets the neural network learn the relationship between the various items that were viewed or purchased in a session.

**LSTM4REC**

During the training phase, the items that are viewed in a session are used to predict the item purchased in the session. Every item is represented as a feature vector of length 64. This analogy corresponds to word embeddings that are used when training LSTM models. The feature vector for every item is obtained using a Variational Autoencoder.

**Generating item embeddings using Variational Auto Encoder:**

  The feature vector for every item is represented by a 904 (number of features) length vector. To mitigate the problem of overfitting and to reduce the complexity of the data representation, Variational Auto Encoder (VAE) is employed for dimensionality reduction of the feature vector. For our use-case, we trained a VAE with a latent space of size 64.**


**Training:**

The LSTM model was trained on a Google Colab instance with 12GB of RAM and NVIDIA Tesla K80 GPU. Sparse Categorical Cross entropy loss function is used with an Adam optimizer. Dropout regularization with an inverted dropout rate of 0.2 is used to avoid overfitting. A validation split of 0.02 is used to evaluate the model at the end of every epoch. If the validation loss does not decrease in four successive epochs the training is automatically stopped.

**Results**
The baseline word2vec model yielded a Mean Reciprocal Rank of 6.4% on the test dataset and the LSTM4REC model gave a Mean Reciprocal Rank of 14.8% on the test dataset.





