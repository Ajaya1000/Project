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





