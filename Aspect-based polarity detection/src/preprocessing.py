import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify.util import accuracy

# Load all tools, lexicon, stop_words and negative words
datadir = "../resources/"
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
neg_list = pd.read_csv(datadir + "negative-words.txt", skiprows=35, header=None)[0].tolist()
pos_list = pd.read_csv(datadir + "positive-words.txt", skiprows=35, header=None)[0].tolist()
stop_words = stopwords.words('english')
neg_words = stop_words[-36:] + [stop_words[116], stop_words[118]] + ['never']
final_stop_words = set(stop_words) - set(neg_words) - set(['more', 'most'])

# Check if the words express high level of emotions
def if_emotional(sentence):
    emotional = ['?!', '!!!', '???', '??', '!!']
    result = False
    for i in emotional:
        if i in sentence:
            result = True
    return result

# Check if the words contain more than two identical letters consecutively
def if_elongated(sentence):
    regex = re.compile(r"(.)\1{2}")
    if sum([1 for word in sentence.split() if regex.search(word)]) > 0:
        return True
    else:
        return False

# Check if all characters in a sentence are uppercase / if all words start from uppercase letters
def if_upper(sentence):
    return (sentence.istitle() or sentence.isupper())

# Check if the list of words contains any positive or negative words from the lexicon
def if_inthelist(sent_list):
    score = 0
    for value in sent_list:
        if value in pos_list:
            score += 1
        if value in neg_list:
            score -= 1
    if score >= 0:
        return True
    else:
        return False

# Check if the list does not contain in either of lexicon lists
def if_notinlists(sent_list):
    score = 0
    for value in sent_list:
        if value in pos_list:
            score += 1
        if value in neg_list:
            score += 1
    if score > 0:
        return False
    else:
        return True

# Define the function making the BOF representation of the
def bag_of_words(words):
    return dict([(word, True) for word in words])

# Define the function generating meaningful bigrams
def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq,n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    concat_bigrams = []
    for tup in bigrams:
        concat_bigrams.append(str(tup[0]+' '+tup[1]))
    return bag_of_words(words + concat_bigrams)

# Preprocess the sentences (tokenizing, applying lowercase, removing stopwords, stemming)
def preprocessor(df):
    feat_list = []
    for index, row in df.iterrows():
        df.loc[index, 'emotion'] = if_emotional(row[4])
        df.loc[index, 'elongated'] = if_elongated(row[4])
        df.loc[index, 'uppercase'] = if_upper(row[4])
        df.loc[index, 'reverse'] = False
        for i in neg_words:
            if i in row[4][:int(row[3].rsplit(':', 1)[0])]:
                df.loc[index, 'reverse'] = True
        row[4] = tokenizer.tokenize(row[4])
        row[4] = [word.lower() for word in row[4]]
        row[4] = [word for word in row[4] if word not in final_stop_words]
        row[4] = [lemmatizer.lemmatize(word) for word in row[4]]
        df.loc[index, 'score'] = if_inthelist(row[4])
        if df.loc[index, 'reverse'] == True:
            df.loc[index, 'score'] *= not df.loc[index, 'score']
        df.loc[index, 'neutral_score'] = if_notinlists(row[4])

        row[4] = bag_of_bigrams_words(row[4])
        row[4]['emotion'] = df.loc[index, 'emotion']
        row[4]['elongated'] = df.loc[index, 'elongated']
        row[4]['uppercase'] = df.loc[index, 'uppercase']
        row[4]['score'] = df.loc[index, 'score']
        row[4]['neutral_score'] = df.loc[index, 'neutral_score']
        row[4]['term'] = True
        feat_list.append(row[4])
    return df, feat_list

# Transform into nltk-compatible format for further classification task
def nltk_compatible(df, featset):
    feat_set = []
    for index, row in df.iterrows():
        feat_set.append((featset[index], row[0]))
    return feat_set

