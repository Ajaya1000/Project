# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 07:08:02 2020

@author: Aju
"""
import string
#Loading a test file into memory
def load_doc(filename):
    #Opening the file as read only
    file=open(filename,'r')
    text= file.read()
    file.close()
    return text

#get all the imgs with their captions
def all_img_captions(filename):
    file=load_doc(filename)
    captions=file.split('\n')
    descriptions ={}
    print("captions[-1] : ",end=':')
    print(captions[-1])
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

#Data cleaning
def cleaning_text(captions):
    table=str.maketrans(" " * len(string.punctuation),string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):
            img_caption.replace("-"," ")
            desc = img_caption.split()
            #converts to lowercase
            
            desc = [word.lower() for word in desc]
            #remove punctuation from each token
            desc  = [word.translate(table) for word in desc]
            #remove hanging 's and a
            desc = [word for word in desc if(word.isalpha())]
            
            #convert back to string
            img_caption= ''.join(desc)
            captions[img][i]=img_caption
    return captions

def text_vocabulary(descriptions):
    #build vocabulary of all unique words
    vocab = set ()
    
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab 

#All description in one file
def save_description(descriptions,filename):
    lines = list ()
    for key,desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key+'\t'+desc)
    data ="\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()
    

    















