#Information retreival(AUT 2020) Assignment #3 TASK #3
#COMPILATION COMMAND - python3 Task3.py ../dataset3/ result.txt
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import string
from sklearn.feature_extraction.text import CountVectorizer ,TfidfTransformer
import sklearn.neighbors
from sklearn import metrics
import sys

dirr = "./dataset3/"
#dirr = sys.argv[1]
V = os.listdir(dirr)

#dictionary declaration for train and test data
train = {}
test = {}
#preprocessing the text
def preprocess(text):
    text = text.lower()             #lower all alphabets 
    email = r'\S+@\S+\.\S+'         #removing email
    url = r'(([hH])ttps?://|www.)\S*'   #removing urls
    spsc=  "#*＊+~-⁃–—_<=>/|\\•・▪■●►◆◈◇◉◊○◯※‘^`'’“„”»«›‹€£$%&§®©™@【】▼▷▶▬≡…√" + '"'  #removing special characters
    punc_table = str.maketrans({key: None for key in string.punctuation+email+url+spsc})
    text=text.translate(punc_table)
    text = word_tokenize(text)    #tokenize the text
    stop = set(stopwords.words('english'))            #removing stopwards
    lemma = nltk.stem.WordNetLemmatizer()           #lemmatization of word
    tokens = [lemma.lemmatize(word) for word in text if word not in stop]
    return tokens

#getting train vocabulary  
def getVocabularytrain(V):
    print("Getting Train vocabulary...\n")
    train['target']=[]  #stores the target attribute as class1 or class2
    train['class1']=[]  #stores words in class 1
    train['class2']=[]  #stores words in class 2
    
    for v in V:
        if v=="class1" or v=="class2":
            textfile_names = os.listdir(dirr + v + "/train/")
            for textfile in textfile_names:
                with open(dirr + v + "/train/" + textfile , encoding="iso-8859-1") as f:
                    text=f.read()  #reading
                    tokens = preprocess(text)  #preprocessing
                    for tok in tokens:
                        train[v].append(tok)
                        if v == 'class1':
                            train['target'].append('1')
                        else:
                            train['target'].append('2')

#getting test vocabulary     
def getVocabularytest(V):
    print("Getting Test vocabulary...\n")
    test['target']=[]   #stores the target attribute as class1 or class2
    test['class1']=[]   #stores words in class 1
    test['class2']=[]   #stores words in class 2
    for v in V:
        if v=="class1" or v=="class2":
            textfile_names = os.listdir(dirr + v + "/test/")
            for textfile in textfile_names:    
                with open(dirr + v + "/test/" + textfile , encoding="iso-8859-1") as f:        
                    text=f.read()   #reading
                    tokens = preprocess(text) #preprocessing
                    for tok in tokens:
                        test[v].append(tok)
                        if v == 'class1':
                            test['target'].append('1')
                        else:
                            test['target'].append('2')
                            
#getting train and test vocab
getVocabularytrain(V)
getVocabularytest(V)
#declaring count vectorizer and tfidftransformer for further use
count_vect= CountVectorizer()
tf_idf=TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
#declaring k value
k_value=[1,10,50]
of = open("result_knn.txt","w") #opening file
of.write("K                1                    10                   50"+"\n")
scor=[]
for i in k_value:
    print("Clssifying for k=",i,".....\n")
    text = train['class1'] + train['class2']
    test_txt= test['class1'] + test['class2']
    #fit and transform train data
    X_train=count_vect.fit_transform(text)
    X_train_tfidf=tf_idf.fit_transform(X_train)
    
    #transform test data
    X_test= count_vect.transform(test_txt)
    X_test_tfidf=tf_idf.fit_transform(X_test)
    
    #fitting the model
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i, weights='distance', n_jobs=10)
    clf.fit(X_train_tfidf, train['target'])
    
    #prediction
    pred = clf.predict(X_test_tfidf)    
    x=metrics.f1_score(test['target'], pred, average='macro')
    
    scor.append(round(x,11))
    
#printing to output file 
of.write("KNN           " + str(scor[0])+ "           " + str(scor[1]) +"             " + str(scor[2]) +"\n")
of.close()
