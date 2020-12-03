#task3
#building inverted index

import os
import sys
import re
import nltk
import json
import pickle

#a tree term set for building b tree for searching
tree_term=set()
#a tokennizer which lower the next and makes tokens out of text
def tokenizer(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]+", " ", text)
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens        

#preprocessing text by tokenizing and stemming using porter stemmer
#used stop words as english 
def preprocessing_txt(text):
    
    tokens = tokenizer(text)
    stemmer = nltk.stem.porter.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    new_text = ""
    for token in tokens:
        token = token.lower()
        if token not in stopwords:
#             print token
            new_text += stemmer.stem(token)
            new_text += " "
        
    return new_text

#inverted index creation
def inverted_index():
    """
    Creates a dictionary of words as key and name of the documents as items
    
    
    "token" : {"doc1":pos1, "doc2":pos2}
    """
    
    inverted = {}
    docs_indexed = 0
    list_doc = os.listdir("./ECT")
    total = len(list_doc)
    point = total / 100
    increment = total / 100
    for doc in list_doc:
#         sys.stdout.write('\r')
        doc_loc = "./ECT/" + str(doc)
        file_doc = open(doc_loc, "r",encoding='utf8')
        file_doc = preprocessing_txt(file_doc.read())
        tokens = tokenizer(file_doc)
        for word in tokens:
            if not inverted.__contains__(word):
                count = 1
                doclist = {}
                doclist[doc] = 1
                inverted[word] = doclist
            else:
                if doc in inverted[word]:
                    doclist = inverted[word]
                    doclist[doc] += 1
                    inverted[word] = doclist
                else:
                    count = 1
                    doclist = inverted[word]
                    doclist[doc] = count
                    inverted[word] = doclist
            tree_term.add(word)        
        docs_indexed += 1
        i = docs_indexed
        if(i % (point) == 0):
            sys.stdout.write("\r[" + "=" * (i / increment) + ">" +  " " * ((total - i)/ increment) + "]" +  str(100*i / float(len(list_doc))) + "%")
            sys.stdout.flush()
    #creting btree
    btree(tree_term)
    return inverted

#utility function for rotating permuterm
def rotate(str, n):
    return str[n:] + str[:n]

#class defination for tr

class TrieNode():
    def __init__(self,key,leaf=False):
        self.key = key   ## key=None for root
        self.childrens = []
        self.leaf = leaf
        self.end = False


class Trie():
    def __init__(self):
        self.root = None

    def insert(self,term):

        
        if self.root == None:
            self.root = TrieNode(None,True)
        
        i = 0
        current = self.root
        prev = self.root
        while i<len(term) and not current.leaf:
            for child in current.childrens:
                if child.key==term[i]:
                    current = child
                    break
            if prev==current:
                break
            else: prev = current
            i+=1

        while i<len(term):
            newNode = TrieNode(term[i],True)
            current.leaf = False
            current.childrens.append(newNode)
            current = newNode
            i+=1

        current.end = True
        
        
    def find_prefix(self,prefix):
        if self.root==None:
            return None

        current = self.root
        prev = self.root
        for ch in prefix:
            if current.leaf:
                return None
            for child in current.childrens:
                if child.key == ch:
                    current = child
                    break
            if prev == current:
                return None
            else: prev = current

        return current
            
    
    def starts_With(self,current,prefix):
        if current==None:
            return []
        if current.leaf:
            return [prefix]    

        terms = []
        if current.end:
            terms.append(prefix)

        for child in current.childrens:
            tmp = self.starts_With(child,prefix+child.key)
            terms = terms+tmp

        return terms

#function to create btree
def btree(tree_term): 
    #object for tree
    trie = Trie()
    for t,term in enumerate(tree_term):
        dkey = term+'$'
        for i in range(len(dkey),0,-1):
            #rotating $ over the term and inserting into btree
            permuterm = rotate(dkey,i)
            trie.insert(permuterm)
        sys.stdout.write("\r{0}/{1} tree terms processed...".format(t+1,len(tree_term)))
        sys.stdout.flush()
    with open("trie.pkl","wb") as f:
        pickle.dump(trie,f)

#crete indexed_docs for the inverted index storing  
if __name__ == "__main__":
    with open("indexed_docs.txt","w") as f:
        indexed_docs = inverted_index()
        f.write(json.dumps(indexed_docs))
   