from sklearn.preprocessing import Normalizer
import numpy as np
from utils import extract_face_roi_single
from Database import addNewStudent
import pickle
import os
from bson.binary import Binary
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='-1'

Normaliser = Normalizer(norm='l2')

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

def get_embedding(face):

    with graph.as_default():
     face=face.astype('float32')
     mean,std=face.mean(),face.std()
     face=(face-mean)/std
     face=np.expand_dims(face,axis=0)
     #embedding=model.predict(face)
     y_pred = graph.get_tensor_by_name("Bottleneck_BatchNorm/cond/Merge:0")
     x= graph.get_tensor_by_name("input_1:0")
     feed_dict = {x: face}
     embedding=sess.run(y_pred,feed_dict)
     return embedding[0]

def get_block_embeddings(path):
     embeddingsArr=[]
     for filename in os.listdir(path):
         img=extract_face_roi_single(os.path.join(path,filename))
         img=get_embedding(img)
         img=np.reshape(img,(-1,2))
         img=Normaliser.transform(img)
         img=np.reshape(img,(128,))
         embeddingsArr.append(img)
     return embeddingsArr

def get_single_embedding(rollno,img,filename):
    img=extract_face_roi_single(img)
    img=get_embedding(img)
    img=np.reshape(img,(-1,2))
    img=Normaliser.transform(img)
    img=np.reshape(img,(128,))
    img=list(img)
    img= Binary(pickle.dumps(img, protocol=2), subtype=128 )
    addNewStudent(rollno,filename,img)




image_path='StudentImages/'
save_path='Embeddings/'

def prepare_data():
    embeddingArr=get_block_embeddings(image_path)
    with open(os.path.join(save_path,'Embeddings.pickle'),'wb') as f:
      pickle.dump((embeddingArr),f)
