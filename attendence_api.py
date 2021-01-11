from flask import request
from flask import Flask,jsonify
import numpy as np
import cv2

from Main import get_attendence
from prepare_embeddings import get_single_embedding
from Database import delete_student

app = Flask(__name__)

@app.route('/getAttendence',methods=['POST'])
def getAttendence():
    data=request.get_json()
    path=data['path']
    methReq=data['methReq']
    if methReq=="Getit":
        cap = cv2.VideoCapture(path)
        ret,img=cap.read()
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        stlist=get_attendence(img)
    return jsonify({"stlist":stlist})

@app.route('/addNewStudent',methods=['POST'])
def addNewStudent():
    data=request.get_json()
    path=data['path']
    name=data['name']
    methReq=data['methReq']
    rollno=data['rollno']
    cap = cv2.VideoCapture(path)
    ret,img=cap.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if methReq=="Add":
     get_single_embedding(rollno,img,name)
     return str('Successfully added the student.')
    else:
        return str('There is some error in adding the embedding.')

@app.route('/deleteStudent',methods=['POST'])
def deleteStudent():
    data=request.get_json()
    rollno=data['rollno']
    methReq=data['methReq']
    if methReq=="delete":
     delete_student(rollno)
     return str("Deleted the student.")

app.run()
