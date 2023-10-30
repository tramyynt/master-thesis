import pandas as pd
import numpy as np
import re
import glob
import re
import nltk
from nltk.tokenize import RegexpTokenizer
import zipfile
import ast
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

liwc2 = pd.read_excel('/home_remote/dic_avg100_annotated_official.xlsx')
liwc2['Terms'] = liwc2['Term'].apply(lambda x: ast.literal_eval(x))
result = dict(zip(liwc2['Category'], liwc2['Terms']))

def parse_dict(_dict):
    '''
    param: dictionary with key as categorie and value as the list of words
    return: a lexicon with key as word and values as the list of categories the word belongs to.
    '''
    lt =[]
    for x in _dict.values():
        lt = lt+x
    mylist = [*set(lt)]
    lexicon = {}
    for i in mylist:
        ls = []
        for j in _dict.keys():
            #print(j)
            if i in _dict[j]:
                ls.append(j)
        lexicon[i] = ls
    return lexicon, list(_dict.keys())

# ref: https://github.com/chbrown/liwc-python
def build_trie(lexicon):
    """
    A `*` indicates a wildcard match.
    """
    trie = {}
    for pattern, category_names in lexicon.items():
        cursor = trie
        for char in pattern:
            if char == "*":
                cursor["*"] = category_names
                break
            if char not in cursor:
                cursor[char] = {}
            cursor = cursor[char]
        cursor["$"] = category_names
    return trie

#ref: https://github.com/chbrown/liwc-python
def search_trie(trie, token, token_i=0):
    if "*" in trie:
        return trie["*"]
    if "$" in trie and token_i == len(token):
        return trie["$"]
    if token_i < len(token):
        char = token[token_i]
        if char in trie:
            return search_trie(trie[char], token, token_i + 1)
    return []

# ref: https://github.com/chbrown/liwc-python
def load_token_parser(dic):
    lexicon, category_names = parse_dict(dic)
    trie = build_trie(lexicon)

    def parse_token(token):
        for category_name in search_trie(trie, token):
            yield category_name

    return parse_token, category_names

def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
    for match in tokenizer.tokenize(text) :
        yield match

def main(input_, result):
   # dic['Terms'] = dic['Term'].apply(lambda x: ast.literal_eval(x))
   # result = dict(zip(dic['Category'], dic['Terms']))
    #lexicon = parse_dict(result)
    parse, category_names = load_token_parser(result)
    input_tokens = tokenize(input_.lower())
    counts = Counter(category for token in input_tokens for category in parse(token))
    
#   a = pd.DataFrame.from_dict(dict(counts),orient = 'index').reset_index()
#   a= a.rename(columns={"index": "Category", 0: "Count"})
#   a['Percentage(%)']= round(a['Count']/len(input_.split(' '))*100,2)
    return counts