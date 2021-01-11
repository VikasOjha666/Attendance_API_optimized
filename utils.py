from mtcnn.mtcnn import MTCNN
import os
import cv2
import numpy as np

def load_image(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=np.asarray(img)
    return img


def extract_face_roi(img_main):
 faceArr=[]
 extractor=MTCNN()
 coordinates=extractor.detect_faces(img_main)
 for i in range(len(coordinates)):
      x1,y1, width, height =coordinates[i]['box']

      x1,y1 = abs(x1), abs(y1)
      x2,y2 = x1+width,y1+height
      face = img_main[y1:y2, x1:x2]
      img=cv2.resize(face,(160,160))
      img=np.asarray(img)
      faceArr.append(img)
 return faceArr


def extract_face_roi_single(img):

 extractor=MTCNN()
 coordinates=extractor.detect_faces(img)
 x1,y1, width, height = coordinates[0]['box']
 x1,y1 = abs(x1), abs(y1)
 x2,y2 = x1+width,y1+height
 face = img[y1:y2, x1:x2]
 img=cv2.resize(face,(160,160))
 img=np.asarray(img)
 return img
