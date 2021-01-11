from pymongo import MongoClient
client=MongoClient('localhost',27017)
db=client.Attendence_database
collection=db.Attendence_collection_db
#collection.insert_one({'Name':'Vikas'})

def addNewStudent(rollno,st_name,img):
    collection.insert_one({'RollNo':rollno,'Name':str(st_name),'Embedding':img})
def list_all_students():
    records=collection.find()
    return records
def delete_student(rollno):
     collection.delete_one({"RollNo":rollno})
