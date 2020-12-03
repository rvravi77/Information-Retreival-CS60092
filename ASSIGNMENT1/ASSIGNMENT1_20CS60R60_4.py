#task4
#wildcard query precess

import pickle
import json
from ASSIGNMENT1_20CS60R60_3 import *
import sys

print("Inverted index loading...")
with open("indexed_docs.txt","r") as f:
    inverted_index=json.load(f)
print("loaded..")
with open("trie.pkl","rb") as f:
    trie = pickle.load(f)


#query process for wild card queried
def query_process(query_term):
    parts = query_term.split('*')
    if len(parts)==1:  #if no wild card is present
        if query_term not in inverted_index.keys():
            return {}
        else:
            return {query_term:inverted_index[query_term]}
    
    if parts[1] == '': ## of type mo*
        case=1
    elif parts[0] == '': ## of type *mon
        case=2
    elif parts[0] != '' and parts[1] != '': ## of type m*n
        case=3
        
    if case == 1:
        query = "$" + parts[0]
    elif case == 2:
        query = parts[1] + "$"
    elif case == 3:
        query = parts[1] + "$" + parts[0]
        
    current = trie.find_prefix(query)
    terms = trie.starts_With(current,query)
    posting_list = {}
    for term in terms:
        term = term.split('$')
        term = term[1]+term[0]  
        posting_list[term] = inverted_index[term]
    return posting_list
    
   

n = len(sys.argv)
if n<2:
    print("query file not given")
    sys.exit()
else:
    query_file = sys.argv[1]


#read line by line from quey.txt
#stores result in txt file with complete posting 
with open(query_file,"r") as f:
    for query_term in f:
        posting_list = query_process(query_term.lower())  ## query each term
        with open("Results1_20CS60R60.txt","w") as f:
            f.write(json.dumps(posting_list))
