import csv
import sys
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
import math
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
STOP_WORDS = stopwords.words("english")
REMOVE_STOPWORDS=True
STEM_WORDS=True
n_words = 0

def write_output_file():
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)

        writer.writerow(["UC1", "L1, L34, L5"]) 
        writer.writerow(["UC2", "L5, L4"]) 

def preprocess(string):
    # Convert words to lower case and clean the text
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)

    # Remove punctuation from text
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)

    # Optionally, remove stop words
    if REMOVE_STOPWORDS:
        string = string.split()
        string = [w for w in string if not w in STOP_WORDS]
        string = " ".join(string)

    # Optionally, shorten words to their stems
    if STEM_WORDS:
          string = string.split()
          stemmer = SnowballStemmer('english')
          stemmed_words = [stemmer.stem(word) for word in string]
          string = " ".join(stemmed_words)

    return string

#Vector representation
def vr(r):
    print(r)
    words = r.split()
    word_count = len(words)
    weights = [0]*n_words

    for i, k in enumerate(inverted):
        if k in words:
            tf = words.count(k)/word_count
            idf = math.log(n/(len(inverted.get(k))),2)
            weights[i] = tf*idf
    return weights

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an argument to indicate which matcher should be used")
        exit(1)

    try:
        match_type = int(sys.argv[1])
    except ValueError as e:
        print("Match type provided is not a valid number")
        exit(1)    

    print(f"Hello world, running with matchtype {match_type}!")

    #GIVEN CODE, REMOVE IT?????
    # Read input low-level requirements and count them (ignore header line).
    #with open("input/low.csv", "r") as inputfile:
      #  print(f"There are {len(inputfile.readlines()) - 1} low-level requirements")

    #Load high and low .csv files into pandas dataframes
    high = pd.read_csv("input/high.csv")
    print(f"There are {len(high)} high-level requirements")
    low = pd.read_csv("input/low.csv")
    print(f"There are {len(low)} low-level requirements")
    
    #Total number of requirements
    n = len(high) + len(low)

    
    #Preprocess text, add to master vocabulary, find frequency
    '''master_vocabulary = []
    tf = [] #frequency of words of the master vocabulary
    d = [] #the number of requirements containing a word of the master vocabulary 
    for df in [high, low]:
        for index, row in df.iterrows():
            r = preprocess(row['text']) #requirement
            df.at[index, 'text'] = r
            #Split words, add them to the master vocabulary, and find their frequencies
            words = r.split()
            words_added = []   
            for word in words:
                if word not in master_vocabulary:
                    master_vocabulary.append(word)
                    tf.append(1)
                else:
                    tf[master_vocabulary.index(word)] += 1
                    #if word not in words_added:
                       # d[master_vocabulary.index(word)] += 1
'''
    #Preprocess text and add it to the list of requirements
    requirements = []
    for df in [high, low]:
        for index, row in df.iterrows():
            r = preprocess(row['text']) #requirement
            df.at[index, 'text'] = r
            requirements.append(r)

    #Inverted index of words, i.e master vocabulary and in which requirements every word is at
    inverted = {}
    for i in range(1,len(requirements)):
        words = requirements[i].split()    
        for word in words:
            inverted.setdefault(word, [])
            if i not in inverted[word]:
                inverted[word].append(i)

    n_words = len(inverted)

    #Vector representation (list containing lists, i.e the vector representation of every requirement)
    vector_representation = [] * n_words
    for r in requirements:
        vector_representation.append(vr(r))
    #print(vector_representation)
    print(inverted)

    #similarity_matrix = [[] * len(low)] * len(high)
    similarity_matrix = [] * len(high)
    for h in range(1, len(high)):
        row = [] * len(low)
        for l in range(1, len(low)):
            vecSize = len(vector_representation[0])
            arrH = np.array(vector_representation[h]).reshape(-1, 1)
            arrL = np.array(vector_representation[len(high) + l]).reshape(-1, 1)

            data = {'high': vector_representation[h],
                    'low': vector_representation[len(high) + l]}
            df = pd.DataFrame (data, columns = ['high','low'])
            row.append(cosine_similarity(df))
            #similarity_matrix[h][l] = cosine_similarity(vector_representation[h], vector_representation[len(high) + l])
        similarity_matrix.append(row)
    print(similarity_matrix)

    write_output_file()