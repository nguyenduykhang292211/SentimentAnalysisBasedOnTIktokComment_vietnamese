from underthesea import word_tokenize, ner
from nltk.stem import *
from nltk import word_tokenize as wt
import pandas as pd
import re
import argparse
import os

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def vietnamese_stemming(text):
    words = wt(text)
    words = [word for word in words if word.isalpha()]
    return ' '.join([token for token in words])

def remove_at(Comment):
    Comment=Comment.split(' ')
    for i in range(len(Comment)):
        if Comment[i].startswith('@'):
            Comment[i]=''
    return ' '.join(Comment)

teencode_df = pd.read_csv('copus/teencode.txt', sep='\t', header=None, names=['teencode', 'replacement'])
teencode_dict = dict(zip(teencode_df['teencode'], teencode_df['replacement']))

def reduce_duplicate_characters(word):
    reduced_word = re.sub(r'(.)\1+', r'\1', word)
    return reduced_word

def token_(Text):
    return word_tokenize(Text, format="text")



def preprocess_without_output(input_file):
    df=pd.read_csv(input_file)
    df['Comment'] = df['Comment'].astype(str)
    df['Comment'] = df['Comment'].apply(remove_at)
    df['Comment'] = df['Comment'].apply(vietnamese_stemming)
    df['Comment'] = df['Comment'].apply(lambda x: ' '.join([teencode_dict.get(word, word) for word in x.split()]))
    df['Comment'] = df['Comment'].str.lower()

    df['Comment'] = df['Comment'].apply(reduce_duplicate_characters)
    df['Comment']=df['Comment'].apply(token_)
    return df

def preprocess(input_file,output_file):
    df=pd.read_csv(input_file)
    df['Comment'] = df['Comment'].astype(str)
    df['Comment'] = df['Comment'].apply(remove_at)
    df['Comment'] = df['Comment'].apply(vietnamese_stemming)
    df['Comment'] = df['Comment'].apply(lambda x: ' '.join([teencode_dict.get(word, word) for word in x.split()]))
    df['Comment'] = df['Comment'].str.lower()

    df['Comment'] = df['Comment'].apply(reduce_duplicate_characters)
    df['Comment']=df['Comment'].apply(token_)
    df.to_csv(output_file,index=False)


    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese text data.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    preprocess(args.input_file, args.output_file)