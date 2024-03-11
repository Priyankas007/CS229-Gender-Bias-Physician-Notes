"""
This file performs the bag of words approach to
distinguish between female and male classes.
"""

import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
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

#
# ## PART 1: import files and separate male/female
# # 1 separating male and female notes
df = pd.read_csv('./processed_data/sections_processed_filtered.csv')
df_f = df[df['1'] == 'F']  # female notes
df_m = df[df['1'] == 'M']  # male notes
#

# print(len(df_f))  # sanity check = 169239
# print(len(df_m))  # sanity check = 161918
#
# # converting section object to list
df_sentences = df['0'].apply(lambda x: sent_tokenize(str(x))).apply(pd.Series, 1).stack()
# # print(df_sentences)
section_list = df['0'].values.tolist()
section_list = [str(item) for item in section_list]
# # print(section_list)
#
# # 2 splitting notes into individual sentences
#tqdm.pandas(desc='processing rows')
#df['0'] = df['0'].progress_apply(lambda x: sent_tokenize(str(x)))
#df_sentences_explode = df.explode('0').reset_index(drop=True)
# print(df_sentences_explode)
#df_sentences_explode.to_csv('sections_sentences.csv', index=False)  # 5045491 entries


# PART 2: preprocessing
# tokenizing (with stemming and lemmatizing)
df = pd.read_csv('train.csv')
section_list = df['text'].values.tolist()
section_list = [str(item) for item in section_list]


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
total_stop_words = list(np.unique(np.array(my_stop_words + standard_stop)))

# print(total_stop_words)
# length = len(section_list)
# tokens = set()
# for i in tqdm(range(len(section_list))):
#     tokens.update(tokenizer_better(section_list[i]))
#     # print(tokens)

# sentence = " ___ hcv cirrhosis c/b ascites, hiv on art, h/o ivdu, copd,  bioplar, ptsd, presented from osh ed with worsening abd  distension over past week.   pt reports self-discontinuing lasix and spirnolactone ___ weeks  ago, because they feels like ""they don't do anything"" and that  they ""doesn't want to put more chemicals in their."" they does not  follow na-restricted diets. in the past week, they notes that they  has been having worsening abd distension and discomfort. they  denies ___ edema, or sob, or orthopnea. they denies f/c/n/v, d/c,  dysuria. they had food poisoning a week ago from eating stale  cake (n/v 20 min after food ingestion), which resolved the same  day. they denies other recent illness or sick contacts. they notes  that they has been noticing gum bleeding while brushing their teeth  in recent weeks. they denies easy bruising, melena, brbpr,  hemetesis, hemoptysis, or hematuria.   because of their abd pain, they went to osh ed and was transferred  to ___ for further care. per ed report, pt has brief period of  confusion - they did not recall the ultrasound or bloodwork at  osh. they denies recent drug use or alcohol use. they denies  feeling confused, but reports that they is forgetful at times.   in the ed, initial vitals were 98.4 70 106/63 16 97%ra   labs notable for alt/ast/ap ___ ___: ___,  tbili1.6, wbc 5k, platelet 77, inr 1.6      "
# print(sent_tokenize(sentence))


# vect = CountVectorizer(tokenizer=tokenizer_better,
#                        max_features=5000,
#                        stop_words=total_stop_words)
#
# # matrix is stored as a sparse matrix (lots of zeros)
# X = vect.fit_transform(section_list)
# print('yes')
# feature_names = list(vect.get_feature_names())
# print(feature_names)
# with open("bow_labels.txt", "w") as txt_file:
#     for line in feature_names:
#         txt_file.write(" ".join(line) + "\n")  # works with any number of elements in a line
# numpyArrayOne = X.toarray()
#
# # Serialization
# numpyData = {"array": numpyArrayOne}
# encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
# print("serialize NumPy array into JSON and write into a file")
# with open("numpyData_bow.json", "w") as write_file:
#     json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
# print("Done writing serialized NumPy array into file")

# print(encodedNumpyData)
# vect.get_feature_names() for word vocabulary
#
# # And make a dataframe out of it
# results = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
# results.to_json(r'bow.json')

"""
"""
tfidf = TfidfVectorizer(tokenizer=tokenizer_better,
                        max_features=10000,
                        stop_words=total_stop_words,
                        use_idf=False,
                        norm='l1')

Y = tfidf.fit_transform(section_list)
print('yes')
tfidf_names = list(tfidf.get_feature_names_out())
print(tfidf_names)
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
# print(encodedNumpyData)


# And make a dataframe out of it
# results2 = pd.DataFrame(Y.toarray(), columns=vect.get_feature_names())
# results2.to_json(r'tfidf.json')

### ~7000 missing values (check)


# 0 pre-processing: remove missing values
# 1 extract section and sex
# 1a remove missing entries
# 2 put into table
# 3 make notes into list of strings


