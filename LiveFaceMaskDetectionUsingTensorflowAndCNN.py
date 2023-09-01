import os
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,Adagrad,RMSprop
from playsound import playsound

train_data_dir = "C:\\Users\\sony\\Downloads\\FaceMask\\facemask-dataset\\dataset"

test_data_dir ="C:\\Users\\sony\\Downloads\\FaceMask\\facemask-dataset\\dataset"

datagen = ImageDataGenerator(
    validation_split=0.40,
    rescale = 1./255
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    color_mode= "grayscale",
    target_size=(48,48),
    batch_size = 32,
    class_mode = "categorical",
    subset = 'training'
)

validation_generator = datagen.flow_from_directory(
    test_data_dir,
    color_mode= "grayscale",
    target_size=(48,48),
    batch_size = 32,
    class_mode = "categorical",
    subset = 'validation'
)


class_label = ['with_mask','without_mask']

print(f'class_label  {len(class_label)}')
print(f'length of label {class_label}')

img,label = train_generator.__next__()

print(img)


# Create a cnn model


model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))

model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(512,activation='relu'))

model.add(Dense(2,activation="softmax"))


model.summary()

import pyttsx3

engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')

print(voices[1].id)

engine.setProperty('voice',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

if __name__ == "__main__":

    speak("Hi Welcome In Image Processing Tutorial of face Mask Detection In Office Using tensorflow . The Compilation of Model will Start Soon I Think You Realy Love Our Hard Work and We Create Multiple Project of Machine Learning and Deep Learning In Future")



model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])


history=model.fit(train_generator,steps_per_epoch=15,epochs=10
                  ,validation_data=validation_generator)


history_df = pd.DataFrame(history.history)

history_df.loc[:,['loss','val_loss']].plot()

history_df.loc[:,['accuracy','val_accuracy']].plot()
#def draw_label(img,text,pos,bg_color):

 #   text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)

  #  exd_x = pos[0] + text_size[0][0] + 2
   # end_y = pos[1] + text_size[0][1] - 2








frame = cv2.imread("C:\\Users\\sony\\Downloads\\Kartik\\Happy\\WIN_20230831_02_08_55_Pro - Copy (2).jpg")

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
labels_dict = {0:'with_mask',1:'without_mask'}
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray,(48,48))
normalize = resized/255.0
reshaped = np.reshape(normalize,(1,48,48,1))

result = model.predict(reshaped)
label = np.argmax(result,axis=1)[0]

print(label)

haar_cascade = cv2.CascadeClassifier('C:\\Users\\sony\\Downloads\\haarcascade_frontalface_default.xml')

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

print(f'Number of faces found={len(faces_rect)}')

for (x,y,w,h) in faces_rect:
 # plt.title('Detection face')

  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)


cv2.imshow("frame",frame)

plt.show()
    


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

if __name__ == "__main__":

    while(True):








        




     

        if (label==0):




            speak("Person wearing a Mask ! You should Giving A permision To Enter In office")
        elif (label==1):






            playsound("C:\\Users\\sony\\Downloads\\emergency-alarm-with-reverb-29431.mp3")
                
            speak("Person Not wearing a Mask ! You should Not  Giving A permision To Enter In office")
    



    