import csv
import sys
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer

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

    # Read input low-level requirements and count them (ignore header line).
    with open("input/low.csv", "r") as inputfile:
        print(f"There are {len(inputfile.readlines()) - 1} low-level requirements")

    nltk.download('stopwords')
    STOP_WORDS = stopwords.words("english")
    REMOVE_STOPWORDS=True
    STEM_WORDS=True
    print(preprocess("the lion is fat and running"))

    write_output_file()