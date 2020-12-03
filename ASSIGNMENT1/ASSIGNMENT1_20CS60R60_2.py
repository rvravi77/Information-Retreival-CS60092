#Task 2
#Building Corpus

import os
import json
import re
import pandas as pd
from bs4 import BeautifulSoup

#function for getting date
def Date(p_tag):
    pos = 0
    month_name = ["January","February","March","April","May","June","July","August","September","October","November","December",""]
    #defining a regular expression for date format
    # a date format is month.{1 or 2 int},{4 int}
    date_format = r"\s*\d{1,2},{0,1}\s*\d{4}|".join(month_name)[:-1]
    
    #finding date in all p_taggraph
    for paragraph in p_tag:
        #finding match using re package in p_taggraph
        the_date = re.findall(date_format,paragraph.text)
        pos=pos+1
        #if date found then break
        if len(the_date)!=0:
            break
        if paragraph.text=="Company Participants":
            break
        if paragraph.text=="Conference Call Participants":
            break
        if paragraph.text=="Operator":
            break
    #returning the date and number of literals data conatains
    return the_date[0],pos

#funtion to get participant list  in Company Participants and Conference Call Participants
def participants_list(p_tag):

    #a list for storing participant names
    participants=[]
    n = 0
    pos = 0
    #iterating over all p_taggraphs
    for paragraph in p_tag:
        pos=pos+1
        #if dour participants encountered then break
        if n==4 or paragraph.text=="Conference Call Participants": 
            break
        #participant list tile contains in string tag
        if paragraph.strong!=None:
            n+=1
            continue
        #removing the detail after '-' i.e removing detail after name
        part_name = paragraph.text.split(' - ')[0]
        participants.append(part_name)
        
    return participants,pos
#function to get presentation text with  key as speakers and value as their statements.
def presentation_nested_dict(p_tag):
    
    name = ""
    said = ""
    pos = 0
    #Declaration for nested dictionary
    presentation = {}
    #interating over all p_taggraphs
    for paragraph in p_tag:
        #finding name of the presentee with strong tag
        if paragraph.strong!=None or paragraph.text!="Question-and-Answer Session":
            if name not in presentation.keys():
                presentation[name] = ""
            presentation[name]+=said
            #break when we have reached the question and answer session in html file
            if paragraph.has_attr('id') and paragraph['id']=="question-answer-session":
                break
            name = paragraph.text
            continue
        said += paragraph.text +" "
        pos =pos+1
    #remove upto the blabk space
    presentation.remove("")
    return presentation,pos

#functiom to find questionaire nested dict
def questionnaire_nested_dictionary(p_tag):
    name = ""
    answer = ""
    questionnaire = {}
    pos = 0
    #interating over all p_tag
    for paragraph in p_tag[1:]:
        #if we found a strong tag
        if paragraph.strong!=None:
            #putting values in nested dictionary
            name = paragraph.text.split(" - ")[-1]
            answer = ""
            questionnaire[pos]={"Name":name ,"Answer":answer}
            pos=pos+1
            answer += paragraph.text+";"
    
    #for adding the last set of speaker and answer
    questionnaire[pos]={"Speaker":name,"Remark":answer}
    questionnaire.remove(0)
    return questionnaire

#create_dict function for dictionary create with attribute date, participant , presentation, Questionnaire
def create_dict(filename):

    with open(filename,'r',encoding='utf8') as f:
        trans_file = f.read()
    soup = BeautifulSoup(trans_file,'html.parser')
    
    p_tag = soup.find_all('p')
    #calling function to extract date
    date,pos = Date(p_tag)
    p_tag = p_tag[pos:]
    #calling function to get participants
    participants,pos = participants_list(p_tag)
    p_tag = p_tag[pos:]
    #calling function to get presentation
    presentation,pos = presentation_nested_dict(p_tag)
    p_tag = p_tag[pos:]
    #calling function to get questionnaire
    questionnaire = questionnaire_nested_dictionary(p_tag)
    
    #combining all the information got
    ect = {"Date":date,"Participants":participants,"Presentation":presentation,"Questionnaire":questionnaire}
    return ect

#a function to text corpus containing text of all html files
def build_text_corpus(filename):
    text = ""
    with open(filename,'r',encoding="utf8") as f:
        ect = f.read()
    soup = BeautifulSoup(ect,'html.parser')
    #whenever a p tag is encountered 
    p_tag = soup.find_all('p')
    #conbine all tet inside p tag
    for paragraph in p_tag:
        text = text+ paragraph.text+" "

    return text


#variables for storing the text(combination of all html files) and 
#dictionary(Dictionary data structure storing all keys for all html files in same txt file)

dictionary_corpus = {}
text_corpus = {}


#getting the directory 
file_dir = "./ECT/"
name_of_file = os.listdir(file_dir)
doc_id=0

for filename in name_of_file:
    try:
        ect = create_dict(file_dir+filename)
    except Exception as e:
        #print(e)
        continue
    dictionary_corpus[doc_id] = ect
    text = build_text_corpus(file_dir+filename)
    with open("./ECTText/"+str(doc_id)+".txt","w") as f:
        f.write(text)
    text_corpus[doc_id] = text
    doc_id+=1


    #created a nested dictionary with given keys and save the set of nested dictionaries as a corpus titled “ECTNestedDict” 
with open("ECTNestedDict.txt","w+") as f:
    f.write(json.dumps(dictionary_corpus))

    #Build a text corpus titled “ECTText” from the set of collected transcripts where each
    #transcript is regenerated as a text file by concatenating all the text information in the transcript
with open("ECTText.txt","w+") as f:
    f.write(json.dumps(text_corpus))
    