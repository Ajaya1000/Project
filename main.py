# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 06:46:20 2020

@author: Aju
"""
import string
import numpy as np
from PIL import Image
import os 
from pickle import dump,load

from keras.applications.xception import Xception,preprocess_input
from keras.preprocessing.image import load_image,img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model,load_model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout

#small libray fro seeing progress of loops
from tqdm import tqdm_notebook as tqdm
tqdm.pandas()  

#set these path according to project folder in you system
dataset_text = "D:\Machine Learning projects\Image_Caption_Generator\Flicker8k_Text"
dataset_images = "D:\Machine Learning projects\Image_Caption_Generator\Flicker8k_Dataset"


