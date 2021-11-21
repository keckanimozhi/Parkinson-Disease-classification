# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:24:39 2021

@author: Administrator
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import cv2
import os
import pickle
from cv2 import *


def load_split(path):
    imagePaths=list(paths.list_images(path))
    data = []
    labels=[]
    for imagePath in imagePaths:
        label=imagePath.split(os.path.sep)[-2]
        image=cv2.imread(imagePath)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image, (200,200))
        image=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features=quantify_image(image)
        data.append(features)
        labels.append(label)
    return (np.array(data),np.array(labels))
def quantify_image(image):
    features=feature.hog(image,orientations=9,
                         pixels_per_cell=(10,10),cells_per_block=(2,2),
                         transform_sqrt=True, block_norm="L1")
    return features
trainingPath =r"D:\csk\Project\Parkinson\dataset\spiral\training"
testingPath=r"D:\csk\Project\Parkinson\dataset\spiral\testing"
(X_train,y_train)=load_split(trainingPath)
(X_test, y_test)=load_split(testingPath)

le = LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)
print(X_train.shape,y_train.shape)
print("[INFO] training model")
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
testingPaths=list(paths.list_images(testingPath))
idxs=np.arange(0,len(testingPaths))
idxs=np.random.choice(idxs,size=(25,),replace=False)
images=[]

for i in idxs:
    image=cv2.imread(testingPaths[i])
    output = image.copy()
    output=cv2.resize(output, (128,128))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200,200))
    image=cv2.threshold(image,0,255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
features = quantify_image(image)
preds = model.predict([features])
label=le.inverse_transform(preds)[0]
color=(0,255,0) if label=="healthy" else (0,0,255)
cv2.putText(output,label,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
images.append(output)
montage=build_montages(images, (128,128), (5,5))[0]
cv2.imshow("Output", montage)
cv2.waitKey(0)
predictions=model.predict(X_test)
cm=confusion_matrix(y_test,predictions).flatten()
print(cm)
(tn,fp,fn,tp)=cm
accuracy=(tp+tn) / float(cm.sum())
print(accuracy)
pickle.dump(model,open('parkinson.pkl','wb'))