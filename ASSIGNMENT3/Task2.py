#Information retreival(AUT 2020) Assignment #3 TASK #2
#COMPILATION COMMAND - python3 Task2.py ../dataset3/ result.txt
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import sys
from sklearn.neighbors import NearestCentroid


dirr="./dataset3/"
#dirr = sys.argv[1]
V = os.listdir(dirr)
#declaring of variables
train = {}  #store train file names
test = {}   #store test file names


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
    return text

#tokenize the text
def tokenize(text):
    text = word_tokenize(text)    #tokenize the text
    stop = set(nltk.corpus.stopwords.words('english'))            #removing stopwards
    lemma = nltk.stem.WordNetLemmatizer()           #lemmatization of word
    tokens = [lemma.lemmatize(word) for word in text if word not in stop]
    return tokens
#get train vocabulary files       
def getVocabularytrain(V):
    train['target']=[]
    train['class1']=[]
    train['class2']=[]
    for f in os.listdir(os.path.join(dirr,'class1','train')):
        train['class1'].append(os.path.join(dirr,'class1','train',f))
        train['target'].append(1)
        
    for f in os.listdir(os.path.join(dirr,'class2','train')):
        train['class2'].append(os.path.join(dirr,'class2','train',f))
        train['target'].append(2)

#get test vocabulary files           
def getVocabularytest(V):
    test['target']=[]
    test['class1']=[]
    test['class2']=[]
    for f in os.listdir(os.path.join(dirr,'class1','test')):
        test['class1'].append(os.path.join(dirr,'class1','test',f))
        test['target'].append(1)
        
    for f in os.listdir(os.path.join(dirr,'class2','test')):
        test['class2'].append(os.path.join(dirr,'class2','test',f))
        test['target'].append(2)
#declarinf tf_idf vectorizer            
tf_idf=TfidfVectorizer(input='filename',encoding='iso-8859-1',decode_error='ignore' ,preprocessor=preprocess,tokenizer=tokenize,norm='l2',use_idf=True,sublinear_tf=True)

#getting train and test vocabulary 
getVocabularytrain(V)
getVocabularytest(V)

#opening the given file
of = open("result2.txt","w")
of.write("b                 0"+"\n")

train_dat= train['class1'] + train['class2']
test_dat=test['class1'] +test['class2']
#Rocchio Classifier
clf = NearestCentroid()
print("Training...")
X_train = tf_idf.fit_transform(train_dat)
clf.fit(X_train, train['target'])

print("Predicting...")
X_test= tf_idf.transform(test_dat)
pred=clf.predict(X_test)

print("Calculating F1 score...")
x = f1_score(test['target'], pred, average='macro')
x= round(x,11)
#writing into o/p file      
of.write("Rocchio      " + str(x) +"\n")
of.close()
