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

#create input sequence pairs from the image description

#data generator, used by model.fit_generator()

def data_generator(descriptions,features,tokenizer,max_length):
    while 1:
        for key,description_list in descriptions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image,input_sequence,output_word = create_sequences(tokenizer,max_length,description_list,feature)
            yield [[input_image,input_sequence],output_word]

def create_sequences(tokenizer,max_length,desc_list,feature):
    X1,X2,y=list(),list(),list()
    #walk through each description for the image
    for desc in desc_list:
        
        #encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        
        #split one sequence into multiple X,y pairs
        
        for i in range(1,len(seq)):
            #split one sequence into input and output pair
            
            in_seq,out_seq = seq[:i],seq[i]
            
            #pad input sequence
            in_seq = pad_sequences([in_seq],maxlen=max_length)[0]
            
            #encode output sequence
            
            out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
            
            #store
            
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
            
    return np.array(X1),np.array(X2),np.array(y)

#You can check the shape of the input and output for your model

[a,b],c = next(data_generator(train_descriptions,features,tokenizer,max_length))
a.shape,b.shape,c.shape
#((47,2048),(47,32),(47,7577))

from keras.utils import plot_model

#define the captioning model
def define_model(vocab_size,max_length):
    
    #features from the CNN model squezzed from 2048 to 256 nodes
    inputs1 =Input(shape=(2048,))
    fe1 =Dropout(0.5)(inputs1)
    fe2 = Dense(256,activation='relu')(fe1)

    #LSTM sequence model
    inputs2 = Input(shape= (max_length,))
    se1 = Embedding(vocab_size,256,mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3=LSTM(256)(se2)
    
    
    #merging both models
    decoder1 = add([fe2,se3])
    decoder2 = Dense(256,activation= 'relu')(decoder1)
    outputs = Dense(vocab_size,activation='softmax')(decoder2)
    
    #tie it together [image,seq] [word]
    model =Model(inputs=[inputs1,inputs2],outputs=outputs)
    
    model.compile(loss ='categorical_crossentropy',optimizer='adam')
    
    #summarize model
    
    print(model.summary())
    
    plot_model(model,to_file='model.png',show_shapes=True)
    
    return model



#train our model
print('Dataset: ',len(train_imgs))
print('Description: train=',len(train_descriptions))
print('Photos: train=',len(train_features))
print('Vocabulary Size:',vocab_size)
print('Description Length: ',max_length)


model = define_model(vocab_size,max_length)
epochs = 10

steps = len(train_descriptions)
#making a directory moels to save out models
os.mkdir("models")
for i in range(epochs):
    generator = data_generator(train_descriptions,train_features,tokenizer,max_length)
    
    model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
    
    model.save("models/model_"+str(i)+".h5")
















