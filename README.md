# tweet_sentiment_extraction_kaggle_competition

The goal of this repository is to explain the approach taken to participate in a Kaggle NLP competition. This repository also contains the Notebooks that were created to compete in [an NLP Kaggle competition](https://www.kaggle.com/c/tweet-sentiment-extraction/overview).

## Competition description

The input dataset is called Sentiment Analysis. The training dataset contains text tweets and emotion (Positive, Negative, or Neutral).  Given the actual tweet and emotion attached to the tweet, the goal was to figure out the word or phrase that supports the tweet the best. More detail available at - https://www.kaggle.com/c/tweet-sentiment-extraction/overview

### Approach taken to solve the problem

At a high level, here is the approach that was taken to solve the problem -

#### 1. Understand the data ([training_data_analysis.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/training_data_analysis.ipynb))
Based upon the analysis, selected word or phrase is primarly the same for all neutral tweets. For positive and negative emotion tweets, selected text is small. So, a submission was made to Kaggle with selected text as original tweet. This gave a baseline score of 0.59451.

#### 2. Look at the text to understand number of words and characters
Look at the words and characters in the tweets to understand more about them. There are some random Unicode characters, a lot of misspelled words, and several slangs in the tweets. This understanding helped design the models better later on. Details at [twitter_text_data_analysis.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/twitter_text_data_analysis.ipynb).

#### 3. Experiment with text pre-processing and transformation
Some experiments to understand what's the difference between the tweet and selected text. Also, if test dataset tweets match well with the training dataset tweets. Regex based tokenization (to make sure that punctutations and special characters are seperated out from the actual word), Lemmetization and Spell correction were experimented with. More details available at [transform_experiments.ipynb] (https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/transform_experiments.ipynb)

Most of the techniques used during this experiment were not used. Transformers turned out the best options to solve this problem. The transformers used subword tokenizers that automatically took care of incomplete or long words. But, this exercise was really good to understand the data and figure out the right solution.

#### 4. Visualize positive, neutral and Negative words
- A embedding is trained using embedding bag (PyTorch) algorithm to understand positive and negative words -  [embedding_bag_model.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/embedding_bag_model.ipynb)

- Visualize positive, neutral and negative words using dimensionality reduction techniques (such as PCA and T-sne) and clustering (with k-means and GMM algorithms) - [pca_and_t_sne_embedding_calculations_and_visualization.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/pca_and_t_sne_embedding_calculations_and_visualization.ipynb)

- Visualize words of positive and negative emotion tweets separately - [visualize_postivie_and_negative_sentences.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/visualize_postivie_and_negative_sentences.ipynb)

Through these visualizations, it was clear that negative and positive words are intermingled in tweets with positive and negative emotions. So, it may be useful to use algorithms that look at whole sentences (e.g. LSTM, GRU, Attention, Transformer, etc.) rather than single word separately.

#### 5. bi-lstm model

Using Glove ("glove.twitter.27B.100d") vector embeddings, a bi-lstm pytorch model was built to predict selected text.

- input tweets were pre-processed to make them ready for bi-lstm PyTorch model - [Data_preprocessing_for_biLSTM.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/bilstm_pytorch/Data_preprocessing_for_biLSTM.ipynb)

- bi-lstm model was run to predict the selected text - [biLSTM_pytorch_model.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/bilstm_pytorch/biLSTM_pytorch_model.ipynb)

bi-lstm model performed better than baseline with Jaccard score of 0.61. But, it's still not close to the leaderboard scores.So, a sequence-to-sequence model with attention was tried.

#### 6. Sequence-to-sequence model with attention

The Sequence-to-sequence model produced results that were very close to the bi-lstm model. At this point, through my research and the guidance from Kaggle grandmasters, I realized that transformers might be the best approach to solve this problem.  


#### 7. "Attention is all you need" transformers

Bert, Roberta, Xlnet, Xlm-roberta, and Electra models were built. All the models performed way better than bi-lstm and sequeuence-to-sequence attention models.  Roberta and Xlnet performed the best with giving a Jaccard score close to 0.715. Detailed implementation available at -
1. [PyTorch Transformers] (https://github.com/sangeetsaurabh/tweet_phrase_kaggle_competition/tree/master/pytorch_transformer)

2. [Keras Transformers] (https://github.com/sangeetsaurabh/tweet_phrase_kaggle_competition/tree/master/keras_transformer)


### Conclusion

With 0.715 scores, this implementation reached quite close to leaders (the winner was 0.735, and the bronze medal score was 0.717). When I started this implementation, only two weeks were left for the competition. Transformers are game-changing - it was great to see that so much can be done just in 2 weeks of part-time implementation. Wish there was a little more time to test and improved the models more.  





