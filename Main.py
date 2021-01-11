import cv2
import numpy as np
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from utils import extract_face_roi
import pickle
from scipy.spatial.distance import cosine
import os
from Database import list_all_students
from flask import request
from flask import Flask,jsonify
from utils import load_image
import tensorflow as tf
import keras



global graph

frozen_graph="facenet_optimized.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
 
with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def,
                          input_map=None,
                          return_elements=None,
                          name="")
sess= tf.Session(graph=graph)



app = Flask(__name__)
def match_faces(emb1,emb2):
    #Matches two faces by distance.
    score=cosine(emb1,emb2)
    if score<0.45:
        return 1
    else:
        return 0


def get_embedding(face):

    with graph.as_default():
     face=face.astype('float32')
     mean,std=face.mean(),face.std()
     face=(face-mean)/std
     face=np.expand_dims(face,axis=0)
     y_pred = graph.get_tensor_by_name("Bottleneck_BatchNorm/cond/Merge:0")
     x= graph.get_tensor_by_name("input_1:0")
     feed_dict = {x: face}
     embedding=sess.run(y_pred,feed_dict)
     #embedding=model.predict(face)

     return embedding[0]


Normaliser = Normalizer(norm='l2')

def load_database_faces(em_path,filename):
    #Loads the embedding of each face.
    with open(os.path.join(em_path,str(filename)),"rb") as f:
        embeddingArr=pickle.load(f)
    return embeddingArr



def get_attendence(img):
    faceArr=extract_face_roi(img)
    allembeddings=[[Each['RollNo'],Each['Name'],Each['Embedding']] for Each in list_all_students()]
    matches=[]
    for face in faceArr:
     face=get_embedding(face)
     face=np.reshape(face,(-1,2))
     face=Normaliser.transform(face)
     face=np.reshape(face,(128,))



     for i in range(len(allembeddings)):
         isMatched=match_faces(face,pickle.loads(allembeddings[i][2]))

         if isMatched==1:
             matches.append((str(allembeddings[i][0]),str(allembeddings[i][1])))
             break


    return matches
