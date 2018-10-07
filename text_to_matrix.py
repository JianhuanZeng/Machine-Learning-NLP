######################
# the learning lab
# original code from https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb
######################

# Download NLTK model data
nltk.download()
nltk.download("book")

vocabulary_size = 300
filename = 'comments_example.csv'

def textprocess(filename, vocabulary_size):
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    
    # Read the data
    print("Reading CSV file...")
    with open(filename, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ["%s,%s,%s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Count the word frequencies
    tokenized_sentences = [nltk.word_tokenize(x) for x in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
    
    # Replace unfrequent words
    for i, x in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in x]

    print('\n Example sentence: %s' % sentences[0])
    print('\n Example sentence after processing: %s' % okenized_sentences[0])

    # Transfer to clean train data
    X_tr = np.asarray([[word_to_index[w] for w in x[:-1]] for x in tokenized_sentences])
    y_tr = np.asarray([[word_to_index[w] for w in x[1:]] for x in tokenized_sentences])

    return  X_tr, y_tr

# print an example
X_tr, y_tr = textprocess(filename, vocabulary_size)
x_example, y_example = X_tr[18], y_tr[18]
print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print('y:\n%s\n%s' % (" ".join([index_to_word[x] for x in x_example]), y_example))
