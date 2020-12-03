#Information retreival(AUT 2020) Assignment #3 TASK #1
 #COMPILATION COMMAND - python3 Task1.py ../dataset3 result.txt
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import string
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import sys
from collections import Counter

#dirr = "../dataset3/"
dirr = sys.argv[1]
V = os.listdir(dirr)
#Varaibles for test and train 
train = {} #for path
test = {}  #for path
docs_word_count = {}  #for mutual information

#preprocess the test
def preprocess(text):
    text = text.lower()             #lower all alphabets 
    email = r'\S+@\S+\.\S+'
    url = r'(([hH])ttps?://|www.)\S*'
    spsc=  "#*＊+~-⁃–—_<=>/|\\•・▪■●►◆◈◇◉◊○◯※‘^`'’“„”»«›‹€£$%&§®©™@【】▼▷▶▬≡…√" + '"'
    for symbol in spsc:
        text = text.replace(symbol, " ")
    clean_txt = str.maketrans({key: None for key in string.punctuation+email+url})
    text=text.translate(clean_txt)
    text = word_tokenize(text)    #tokenize the text
    stop = set(nltk.corpus.stopwords.words('english'))            #removing stopwards
    lemma = nltk.stem.WordNetLemmatizer()           #lemmatization of word
    tokens = [lemma.lemmatize(word) for word in text if word not in stop]
    return tokens
#get train vocabulary files       
def getVocabularytrain(V):
    train['target']=[]
    train[1]=[]
    train[2]=[]
    c1t=[]
    c2t=[]
    for f in os.listdir(os.path.join(dirr,'class1','train')):
        c1t.append(os.path.join(dirr,'class1','train',f))
    for f in os.listdir(os.path.join(dirr,'class2','train')):
        c2t.append(os.path.join(dirr,'class2','train',f))
    for file in c1t:
        with open(file,"r",encoding='iso-8859-1', errors='ignore') as f:
            text = f.read()
        tokens = preprocess(text)
        for t in tokens:
            train[1].append(t) #storing for classification
            train['target'].append(1) 
        tokensCount = Counter(tokens)
        for term,doc_freq in tokensCount.items():  
            if term not in docs_word_count.keys():
                docs_word_count[term] = [0,0]  
            docs_word_count[term][0] += 1   #calculating word count in class
        
    for file in c2t:
        with open(file,"r",encoding='iso-8859-1', errors='ignore') as f:
            text = f.read()
        tokens = preprocess(text)
        for t in tokens:
            train[2].append(t) #storing for classification
            train['target'].append(2) 
        tokensCount = Counter(tokens)
        for term,doc_freq in tokensCount.items():  
            if term not in docs_word_count.keys():
                docs_word_count[term] = [0,0]  
            docs_word_count[term][1] += 1   #calculating word count in class
        
#get test vocabulary files           
def getVocabularytest(V):
    test['target']=[]
    test[1]=[]
    test[2]=[]
    c1t=[]
    c2t=[]
    for f in os.listdir(os.path.join(dirr,'class1','test')):
        c1t.append(os.path.join(dirr,'class1','test',f))
    for f in os.listdir(os.path.join(dirr,'class2','test')):
        c2t.append(os.path.join(dirr,'class2','test',f))
    for file in c1t:
        with open(file,"r",encoding='iso-8859-1', errors='ignore') as f:
             text = f.read()
        tokens = preprocess(text)
        for t in tokens:
             test[1].append(t)
             test['target'].append(1) #store classs of a word
        
    for file in c2t:
        with open(file,"r",encoding='iso-8859-1', errors='ignore') as f:
             text = f.read()
        tokens = preprocess(text)
        for t in tokens:
             test[2].append(t)
             test['target'].append(2) #store classs of a word

#N11: docs containing word and in class; 
#N10: docs containing word and not in class; 
#N00: docs not containing word, and not in class; 
#N01: docs not containing word and in the class.      
def mutual_info_score(word):
    
    N1 = len(train[1])
    N2 = len(train[2])
    N11 = docs_word_count[word][0]
    N10 = docs_word_count[word][1]
    N01 = N1 - docs_word_count[word][0]
    N00 = N2 - docs_word_count[word][1]
   
    N1_dot = N11 + N10
    N0_dot = N01 + N00
    Ndot_1 = N11 + N01
    Ndot_0 = N10 + N00
    N = N10 + N11 + N01 + N00
    #mutual info calculation
    if N10 > 0 and N1_dot>0:
        third_log = (N10/N) * math.log((N*N10)/(N1_dot*Ndot_0), 2)
    else:
        third_log = 0
    if not N11 > 0:
        return float("-inf")
    if N11 > 0 and N1_dot > 0:
        first_log = (N11/N) * math.log((N*N11)/(N1_dot*Ndot_1), 2)
    else:
        first_log = 0
    try:
        return first_log + (N01/N) * math.log((N*N01)/(N0_dot*Ndot_1), 2) + third_log + (N00/N) * math.log((N*N00)/(N0_dot*Ndot_0), 2)
    except:
        return 0
#getting train and test vocab path
print("Getting training data ... ")                
getVocabularytrain(V)
print("Getting testing data ... ")
getVocabularytest(V)

#combining all training and testing words
X_train= train[1] + train[2]
X_test= test[1] + test[2]
features=[1,10,100,1000,10000]

#of = open("result_20CS60R60_A3T1.txt","w")
#opening file 
of = open(sys.argv[2],"w")
of.write("NumFeature                 1                     10                    100                   1000                   10000"+"\n")
scor=[]  #store F1 score for each classification
#classifying for every given feature

for i in features:
    #calculating mutual info scores
    Features=set()
    print("Classifying for ",i," Features")
    mi = []
    for term in docs_word_count.keys():
        mi_score = mutual_info_score(term)
        mi.append((mi_score,term))
        
    mi.sort(reverse=True)
    #selecting top i features
    Features = set([term for mi_score,term in mi[:i]])
    
    #TfidfVectorizer with vocabulary as Features obtained from mutual information
    tf_idf=TfidfVectorizer(vocabulary=Features,norm='l2',use_idf=True,sublinear_tf=True)

    X_train_tfidf=tf_idf.fit_transform(X_train) #fit and trasform train data
    X_test_tfidf=tf_idf.transform(X_test) #transform test data
    
    clf = MultinomialNB(alpha=1.0) 
    clf2 = BernoulliNB(alpha=1.0)
    clf.fit(X_train_tfidf, train['target'])  #fitting multinomial NB
    clf2.fit(X_train_tfidf, train['target']) #fitting Bernoulli NB
    
    pred = clf.predict(X_test_tfidf) #predicting for Multinomial NB
    pred2 = clf2.predict(X_test_tfidf)#predicting for Bernoulli NB
    
    x=metrics.f1_score(test['target'], pred, average='macro')
    x2=metrics.f1_score(test['target'], pred2, average='macro')
    scor.append(round(x,11))
    scor.append(round(x2,11))

of.write("MultinomialNB          " + str(scor[0])+ "            " + str(scor[2]) +"              " + str(scor[4]) +"            " + str(scor[6]) +"             " + str(scor[8]) +"\n")
of.write("BernoulliNB           " + str(scor[1])+ "             " + str(scor[3]) +"              " + str(scor[5]) +"            " + str(scor[7]) +"             " + str(scor[9]))
of.close()