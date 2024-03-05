"""
This file performs the bag of words approach to
distinguish between female and male classes.
"""

import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tqdm
import json
from json import JSONEncoder

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


### CLASSES ###
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


## PART 1: import files and separate male/female
# 1 separating male and female notes
df = pd.read_csv('./processed_data/sections_processed_filtered.csv')
df_f = df[df['1'] == 'F']  # female notes
df_m = df[df['1'] == 'M']  # male notes

# print(len(df_f))  # sanity check = 169239
# print(len(df_m))  # sanity check = 161918

# converting section object to list
section_list = df['0'].values.tolist()
section_list = [str(item) for item in section_list]
# print(section_list)


# PART 2: preprocessing
# tokenizing (with stemming and lemmatizing)
def tokenizer_better(text):
    # tokenize text by replacing punctuation and numbers with spaces and lowercase all words
    punc_list = string.punctuation + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    words = word_tokenize(text)
    # filter out short words
    words = [word for word in words if len(word) > 1]
    # stem words
    porter_stemmer = PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    # lemmatize words
    lemmer = WordNetLemmatizer()
    words = [lemmer.lemmatize(word) for word in words]
    return words


# stop words
my_stop_words = ['the','and','to','of','was','with','a','on','in','for','name',
                 'is','patient','s','he','at','as','or','one','she','his','her','am',
                 'were','you','pt','pm','by','be','had','your','this','date',
                 'from','there','an','that','p','are','have','has','h','but','o',
                 'namepattern','which','every','also']
standard_stop = stopwords.words("english")
#print(standard_stop)
total_stop_words = list(set(my_stop_words + standard_stop))
print(total_stop_words)

vect = CountVectorizer(tokenizer=tokenizer_better,
                       max_features=5000,
                       stop_words=total_stop_words)
vect.fit(section_list)

# matrix is stored as a sparse matrix (lots of zeros)
X = vect.transform(section_list)
print('yes')
# print(vect.get_feature_names())
#print(np.asarray(X.sum(axis=0)))
numpyArrayOne = X.toarray()

# Serialization
numpyData = {"array": numpyArrayOne}
encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
print("Printing JSON serialized NumPy array")
#print(encodedNumpyData)
# vect.get_feature_names() for word vocabulary

"""
"""
tfidf = TfidfVectorizer(tokenizer=tokenizer_better,
                        max_features=3000,
                        stop_words=my_stop_words)
tfidf.fit(section_list)
Y = tfidf.transform(section_list)
#print(tfidf.get_feature_names())
#print(np.asarray(Y.sum(axis=0)))
tfidf_names = list(tfidf.get_feature_names_out())
#print(tfidf_names)

with open("tfidf_labels_16000.txt", "w") as txt_file:
    for line in tfidf_names:
        txt_file.write(" ".join(line) + "\n")  # works with any number of elements in a line
# print(np.asarray(Y.sum(axis=0)))
numpyArrayTwo = Y.toarray()

# Serialization
numpyData2 = {"array": numpyArrayTwo}
encodedNumpyData2 = json.dumps(numpyData2, cls=NumpyArrayEncoder)  # use dump() to write array into file
print("serialize NumPy array into JSON and write into a file")
with open("numpyData_tfidf_16000.json", "w") as write_file:
    json.dump(numpyData2, write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")

# And make a dataframe out of it
results2 = pd.DataFrame(Y.toarray(), columns=tfidf.get_feature_names())
results2.to_json(r'tfidf.json')

### ~7000 missing values (check)

# 0 pre-processing: remove missing values
# 1 extract section and sex
# 1a remove missing entries
# 2 put into table
# 3 make notes into list of strings


