#Information retreival(CS60092)- AUT 2020
#ASSIGNMENT 2
"""
Compilation command : python3 ASSIGNMENT2_20CS60R60.py query.txt

query.txt is in same directory as of code
assumed html files are in "../Dataset/Dataset/"

StaticQualityScore.pkl is in  "../Dataset/StaticQualityScore.pkl"

Leaders.pkl is in "../Dataset/Leaders.pkl"

"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import pickle5 as pickle
import string
import os
import sys
from collections import Counter
import math
import numpy as np
from collections import defaultdict
from bs4 import BeautifulSoup


directory = "./Dataset/Dataset/"
files = os.listdir(directory)
files.sort(key=lambda filename: int(filename.split(".")[0]))
inverted_index = {} #dictionary to store inverted index
champion_listlocal ={} #dictionary to store championlist_global
champion_listglobal ={} #dictionary to store championlist_global
DF = {}   #document frequency
tf={}     #term frequency
index_champ={}
index_champg={}

sqs=open("./Dataset/StaticQualityScore.pkl",'rb')
sqs_list = pickle.load(sqs)

ldr=open("./Dataset/Leaders.pkl",'rb')
ldr_list=pickle.load(ldr)

#funtion to preprocess text 
def preprocess(text):
    text = text.lower()             #lower all alphabets 
    punc_table = str.maketrans({key: None for key in string.punctuation+"–"+"’"})
    text=text.translate(punc_table)
    text = word_tokenize(text)    #tokenize the text
    stop = set(stopwords.words('english'))            #removing stopwards
    lemma = nltk.stem.WordNetLemmatizer()           #lemmatization of word
    tokens = [lemma.lemmatize(word) for word in text if word not in stop and not isNumeric(word)]
    return tokens
#funtion to check if the text has number or not
def isNumeric(s):
    try:
        float(s)
        return True
    except:
        return False
#funtion to build inverted positional  posting for given file
def build_index(tokens,docId):
    """
    inverted index format
    ----------
    term1 : "idf"     : idf
            "posting" : [[doc1,tf_td1] [doc2,tf_td2] [doc3,tf_td3] ..... ]
            
    term2 : "idf"     : idf
            "posting" : [[doc1,tf_td1] [doc2,tf_td2] [doc3,tf_td3] ..... ]

            .
            .
            .
    
    -------
    """
    for pos,token in enumerate(tokens):
       
        tf_td = tf[token,docId] 
        if (token not in inverted_index.keys()):
            inverted_index[token] = {'posting':[]}
        if ( (docId,tf_td)  not in  inverted_index[token]['posting']):     
            inverted_index[token]['posting'].append((docId,tf_td))
#function to create champion index
def create_champion_index():
    # This method gets executed as soon as inverted index is built
    # This iterates through dictionary and stores top 50 documents with highest tf-idf from posting list
    """
    champion index format
    -------
    term1 : [[doc1,tf_td1] , ..........   [doc50,tf_td50]]
    
    term2 : [[doc1,tf_td1] , ..........   [doc50,tf_td50]]
            .
            .
            .

    """
    
    for token in inverted_index.keys():
        for doc in inverted_index[token]['posting']:
            if token in index_champ.keys():
                index_champ[token].append([doc[0],doc[1]])
            else:
                index_champ[token]=[[doc[0],doc[1]]]
            
    for token in inverted_index.keys():
        for doc in inverted_index[token]['posting']:
            if token in index_champg:
                index_champg[token].append([doc[0],((doc[1]*inverted_index[token]['idf']) + sqs_list[doc[0]])])
            else:
                index_champg[token]=[[doc[0],((doc[1]*inverted_index[token]['idf']) + sqs_list[doc[0]])]]
            
    for term,value in index_champ.items():
        #list_pair = index_champ[term]
        sorted_list_pair = sorted(value, key= lambda list: list[1] , reverse = True)
        champion_listlocal[term] = sorted_list_pair[:50]
        
    for term,value in index_champg.items():
        #list_pair = index_champ[term]
        sorted_list_pair = sorted(value, key= lambda list: list[1] , reverse = True)
        champion_listglobal[term] = sorted_list_pair[:50]
#function to calculate document frequency
def calculate_df():
    for i in DF:
        DF[i]=len(DF[i])
#function to create inverted positional index for every file
def ipv():
    for file in files:
       
        docId = int(file.split(".")[0])
        with open(directory+file,"r",encoding="utf8") as f:
            text2 = f.read()
        soup = BeautifulSoup(text2,'html.parser')
        text=soup.getText()
        tokens = preprocess(text)
        for w in tokens:
            try:
                DF[w].add(docId)
            except:
                DF[w] = {docId}
        
        counter=Counter(tokens)
        for token in np.unique(tokens):
            tf[token,docId] = math.log10(1+counter[token])
        build_index(tokens,docId)
        sys.stdout.write("\r{0}/{1} files indexed...,{2}>".format(docId+1,len(files),"="*(docId//100)))
        sys.stdout.flush()
#function to compute the  normalized doc length       
def compute_doc_lengths(index):
    """
        Return a dict mapping doc_id to length, computed as sqrt(sum(tf-idf**2)),
        
    """
    result = defaultdict(lambda: 0)
    for val in index:
        for doc in index[val]['posting']:
            tf_idf=index[val]['idf'] * doc[1]
            result[doc[0]] += math.pow(tf_idf,2)
    for key, value in result.items():
        result[key] = math.sqrt(result[key])
    return result
    pass
#function to convert query terms to an vector
def query_to_vector(query_terms):
    """ Convert a list of query terms into a dict mapping term to inverse document frequency.
	using log(N / DF(term))
    """
    
    query_doc_freq = {}
    for word in query_terms:
        try:
            query_doc_freq[word]=DF[word]
        except:
            continue
    
    result = {}
    for key in query_doc_freq.keys():
        result[key] = math.log((1000/query_doc_freq[key]),10) 
    return result

    pass
#function to search by cosine similarity
def search_by_cosine(query_vector, index, doc_lengths):
    """
    Return a sorted list of doc_id, score pairs, where the score is the
        cosine similarity between the query_vector and the document
    """
    scores = defaultdict(lambda: 0)
    for query_term, query_weight in query_vector.items():
        for doc_id in index[query_term]['posting']:
            scores[doc_id[0]] += query_weight * doc_id[1]  
    for doc_id in scores:
        scores[doc_id] /= doc_lengths[doc_id]
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    pass
#function to search by champion list score
def search_by_champ_loc(query_vector, index, doc_lengths):
    
    scores = defaultdict(lambda: 0)
    for query_term, query_weight in query_vector.items():
        for doc_id, doc_weight in index[query_term]:
            scores[doc_id] += query_weight * doc_weight  
    for doc_id in scores:
        scores[doc_id] /= doc_lengths[doc_id]
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    pass
#functio to search with Cluster Prunning Scheme
def search_by_leader(query_vector, index, doc_lengths):
    scores = defaultdict(lambda: 0)
    for query_term, query_weight in query_vector.items():
        for doc_id in index[query_term]['posting']:
            if doc_id[0] in ldr_list:scores[doc_id[0]] += query_weight * doc_id[1]  
    for doc_id in scores:
        scores[doc_id] /= doc_lengths[doc_id]
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    pass
#passing the qury file
n = len(sys.argv)
if n<2:
    query_file = "query.txt"
else:
    query_file = sys.argv[1]    

def main():
    
    print("calculating inverted pov Index ...")    
    ipv() 
    print("\ncalculating DF ...")
    calculate_df()
    for token in inverted_index.keys():
        idf =  math.log10((1000/DF[token])) 
        inverted_index[token]['idf']=idf
    print("creating champion index(global & local) ...\n") 
    create_champion_index()
     
    doc_lengths=compute_doc_lengths(inverted_index)
    of = open("RESULTS2_20CS60R60.txt","w")
    print("start Quering ..",flush=True)
    with open(query_file,"r") as f:
        for query_term in f:
            print("calculating scores for query :",query_term)
            
            #preprocess the query
            stemmed_query = preprocess(query_term)
            query_vec = query_to_vector(stemmed_query)
            #tf idf score
            result_docIds1 = []
            result_docIds1 = search_by_cosine(query_vec, inverted_index, doc_lengths)
            s1= result_docIds1[:10]
            listToStr1 = ' '.join(map(str, s1))
            #Local Champion List Score
            result_docIds2 =[]
            result_docIds2 = search_by_champ_loc(query_vec, champion_listlocal, doc_lengths)
            s2= result_docIds2[:10]
            listToStr2 = ' '.join(map(str, s2))
            #Global Champion List Score
            result_docIds3 =[]
            result_docIds3 = search_by_champ_loc(query_vec, champion_listglobal, doc_lengths)
            s3= result_docIds3[:10]
            listToStr3 = ' '.join(map(str, s3))
            #– Cluster Prunning Scheme
            result_docIds4 =[]
            result_docIds4 = search_by_leader(query_vec,inverted_index, doc_lengths)
            s4= result_docIds4[:10]
            listToStr4 = ' '.join(map(str, s4))
            
            #combining all into a string to write into file
            listToStr='\n' + query_term + '\n' +listToStr1 + '\n' + listToStr2 + '\n' + listToStr3 +'\n' +listToStr4 + '\n'
           
            of.write(listToStr)
    of.close()
    print("done ...")

if __name__ == "__main__":
    main()
