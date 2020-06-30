# tweet_sentiment_extraction

The goal of this repository is to explain the approach that was taken to participate in a Kaggle NLP competition. This repository also contains the Notebooks that were created to compete in a NLP Kaggle competition (https://www.kaggle.com/c/tweet-sentiment-extraction/overview).

## Competition description

The input dataset is called Sentiment Analysis. The training dataset contains text tweets and emotion (Positive, Negative, or Neutral).  Given the actual tweet and emotion attached to the tweet, the goal was to figure out the word or phrase that supports the tweet the best. More detail available at - https://www.kaggle.com/c/tweet-sentiment-extraction/overview

### Approach taken to solve the problem

At a high level, here is the approach that was taken to solve the problem -

#### 1. Understand the data ([training_data_analysis.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/training_data_analysis.ipynb))
Based upon the analysis, selected word or phrase is primarly the same for all neutral tweets. For positive and negative emotions, selected text is small. So, I went ahead and made a submission to Kaggle with selected text as text. This gave me a baseline score of 0.59451.

#### 2. Look at the text to understand number of words and characters
Look at the words and characters to understand more about them. There are some random Unicode characters, a lot of misspelled words, and several slangs in the tweets. This understanding will help design the models better later on. Details at [twitter_text_data_analysis.ipynb](https://github.com/sangeetsaurabh/tweet_sentiment_extraction/blob/master/twitter_text_data_analysis.ipynb).

#### 3. Text pre-processing and post-processing experiment

