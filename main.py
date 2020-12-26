# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 06:46:20 2020

@author: Aju
"""
from utility import *
import string
import numpy as np
from PIL import Image
import os 
from pickle import dump,load

from keras.applications.xception import Xception,preprocess_input
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model,load_model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout

#small libray fro seeing progress of loops
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()  

#set these path according to project folder in you system
dataset_text = "D:\Machine Learning projects\Image_Caption_Generator\Flicker8k_Text"
dataset_images = "D:\Machine Learning projects\Image_Caption_Generator\Flicker8k_Dataset"


#We prepare out text data
filename = dataset_text+ "/" + "Flickr8k.token.txt"
#loading the file that contains all data
#mapping them into descripton dictionary img to 5 captions
descriptions = all_img_captions(filename)
print("Lenght of description=", len(descriptions))

#cleaning the descriptions
clean_descriptions = cleaning_text(descriptions)

#building vocabulary
vocabulary = text_vocabulary(clean_descriptions)
print("Lenght of vocabulary= ",len(vocabulary))
#saving each description to file
save_description(clean_descriptions,"descriptions.txt")
"""
def extract_features(directory):
    model = Xception(include_top=False,pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory+"/"+img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image,axis=0)
        image = image/127.5
        image = image - 1.0
        
        feature = model.predict(image)
        features[img] = feature
    return features

features = extract_features(dataset_images)
dump(features, open("features.p","wb"))
"""
features = load(open("features.p","rb"))

filename = dataset_text + "/"+"Flickr_8k.trainImages.txt"

train_imgs = load_photos(filename)
train_descriptions = load_clean_description("descriptions.txt",train_imgs)

train_features = load_features(train_imgs)

#give each word an index , and store that into
#tokenizer.p pickle file

tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer,open('tokenizer.p','wb'))

vocab_size = len(tokenizer.word_index) + 1
vocab_size

max_length = max_length(descriptions)
max_length



