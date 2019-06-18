import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re, itertools
import nltk
from sklearn.metrics import classification_report,roc_curve,auc
from sklearn.preprocessing import OneHotEncoder

# Preprocesses the raw data as required
def record_process(someposts, features = ['title', 'selftext']):
    #1.Exclude rows from the training set where `selftext`
    someposts.drop(someposts[someposts['selftext']==''].index, inplace=True)
    someposts.drop(someposts[someposts['selftext']=='[deleted]'].index, inplace=True)
    someposts.drop(someposts[someposts['selftext']=='[removed]'].index, inplace=True)

    #2. Exclude rows with less than `5` comments.
    someposts = someposts.loc[someposts['num_comments']>4]
    print('There are {} records after processing'.format(len(someposts)))

    #3. Only use the `title` and `selftext` fields as a source of features.
    # simply merge them with start and end token later

    #4. Make a decision on how to handle subreddit categories with fewer than 1000 examples:
    # simply merge them into one
    # because the sum amount of rare categories is about 1142, a small amount.
    subreddit_distribution = someposts['subreddit'].value_counts()
    rare_categories = subreddit_distribution[subreddit_distribution<1000].index
    someposts['subreddit'][someposts['subreddit'].isin(rare_categories)] = 'others'
    print('The sum of rare categories is',subreddit_distribution[subreddit_distribution<1000].sum())

    # 5. map labels
    le = LabelEncoder()
    someposts['target'] = le.fit_transform(someposts['subreddit'])
    subreddit_mappings = {index: label for index, label in enumerate(le.classes_)}

    return subreddit_mappings, someposts[features+['target']]

# remove punctuation and do stemming
def process_sentence(sentence):
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    ps = nltk.stem.porter.PorterStemmer()
    sentence = [ps.stem(word) for word in sentence] # if not word in set(stopwords.words('english'))
    return sentence

# process text as [1,21,3...]
def textprocess(sentences, vocabulary_size):
    unknown_token = "UNKNOWN_TOKEN"

    # Count the word frequencies
    tokenized_sentences = [process_sentence(x) for x in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    for i, sentence in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sentence]

    # transfer to clean train data
    X = np.asarray([[word_to_index[w] for w in x] for x in tokenized_sentences])

    try:
        print('\nExample sentence: %s' % sentences[0])
        print('\nExample sentence after processing: %s' % tokenized_sentences[0])
        print('\nExample input sentence: %s' % X[0])
    except:
        print()

    return index_to_word, X

# partitions the model-ready data into train, validation, and test sets.
def partition_dataset(someposts, data):
    assert someposts['target'].isnull().sum()==0, 'there is target with null value'
    assert len(data)==len(someposts['target']), 'unmatching input and output'
    X_train, X_test, y_train, y_test = train_test_split(data, someposts['target'], test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, X_test, X_val, y_train, y_test, y_val
