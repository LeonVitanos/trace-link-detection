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

def write_output_file(link):
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)
        for h in range(len(link)):
            lnk = ""
            for l in range(1, len(link[h])):
                if lnk == "":
                    lnk = link[h][l]
                else:
                    lnk = lnk +  "," + link[h][l]
            writer.writerow([link[h][0], lnk])

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
    #print(r)
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
    #print(f"{high.at[0, 'text']}")
    #print(f"{low.at[0, 'text']}")
    for df in [high, low]:
        for index, row in df.iterrows():
            r = preprocess(row['text']) #requirement
            df.at[index, 'text'] = r
            requirements.append(r)

    #print(f"{requirements[0]}")
    #print(f"{requirements[len(high)]}")

    #Inverted index of words, i.e master vocabulary and in which requirements every word is at
    inverted = {}
    for i in range(0,len(requirements)):
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
    #print(f"{vector_representation[0]}")
    #print(f"{vector_representation[len(high)]}")
    #print(inverted)

    
    similarity_matrix = [] * len(high)
    trace_link = [] * len(high)
    for h in range(len(high)):
        row = [] * len(low)
        link_row = [high.at[h, 'id']]
        for l in range(0, len(low)):
            arrH = np.array(vector_representation[h]).reshape(1, -1)
            arrL = np.array(vector_representation[len(high) + l]).reshape(1, -1)
            sim = cosine_similarity(arrH, arrL)[0][0]
            row.append(sim)
            if match_type == 0 and sim > 0:
                link_row.append(low.at[l, 'id'])
            elif match_type == 1 and sim >= 0.25:
                link_row.append(low.at[l, 'id'])
        similarity_matrix.append(row)
        trace_link.append(link_row)
        if match_type == 2:
            maxSim=max(similarity_matrix[h])
            for l in range(len(low)):
                    if similarity_matrix[h][l] >= 0.67 * maxSim:
                        trace_link[h].append(low.at[l, 'id'])
        elif match_type == 3:
            maxSim=max(similarity_matrix[h])
            for l in range(len(low)):
                    if similarity_matrix[h][l] >= 0.67 * maxSim and similarity_matrix[h][l] >= 0.25:
                        trace_link[h].append(low.at[l, 'id'])

    '''
    if match_type == 2:
        for h in range(len(high)):
            for l in range(len(low)):
                if similarity_matrix[h][l] >= 0.67 * maxSim:
                    trace_link[h].append(low.at[l, 'id'])
    '''

    #print(trace_link)

    write_output_file(trace_link)

    TP = 0 # True Positive, trace link manually identified and predicted by tool
    FP = 0 # False Positive, trace link not manually identified, but predicted by tool
    FN = 0 # False Negative, trace link manually identified, but not predicted by tool
    TN = 0 # True Negative, trace link not manually identified and not predicted by tool

    links = pd.read_csv("input/links.csv")
    manual = []
    for h in range(len(links)):
        string = links.at[h, 'links']
        row = []
        if isinstance(string, str):
            appending = False
            number = ""
            for ch in string:
                if ch.isdigit():
                    number += ch
                    appending = True
                if not ch.isdigit() and appending:
                    row.append(int(number))
                    number = ""
                    appending = False
            row.append(int(number))
        manual.append(row)

    links = pd.read_csv("output/links.csv")
    predict = []
    for h in range(len(links)):
        string = links.at[h, 'links']
        row = []
        if isinstance(string, str):
            appending = False
            number = ""
            for ch in string:
                if ch.isdigit():
                    number += ch
                    appending = True
                if not ch.isdigit() and appending:
                    row.append(int(number))
                    number = ""
                    appending = False
            row.append(int(number))
        predict.append(row)

    for h in range(len(manual)):
        manLinks = manual[h]
        preLinks = predict[h]
        i = 0
        j = 0
        while i < len(manLinks) and j < len(preLinks):
            if manLinks[i] == preLinks[j]:
                TP += 1
                i += 1
                j += 1
            else:
                if manLinks[i] < preLinks[j]:
                    FN += 1
                    i += 1
                else:
                    FP += 1
                    j += 1
        while i < len(manLinks):
            FN += 1
            i += 1
        while j < len(preLinks):
            FP += 1
            j += 1
    TN = len(high) * len(low) - (TP + FN + FP)

    print(f"True Positives: {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"True Negatives: {TN}")
