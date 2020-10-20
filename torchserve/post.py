import requests
import json

fn = '../data/multiface.jpg'
#data = {'threshold' : '0.7', 'imgurl':'https://datahunt.s3.ap-northeast-2.amazonaws.com/uploads/3/pm_3c7srelc9awe.jpg'}
#data = {'threshold' : '0.7', 'imgurl':'https://datahunt.s3.ap-northeast-2.amazonaws.com/uploads/37/10143978_1566774062_image1_L.jpg'}
#data = {'threshold' : '0.7', 'imgurl':'https://datahunt.s3.ap-northeast-2.amazonaws.com/uploads/37/10016369_1565766485_image1_L.jpg', 'req_image' : 'True'}
data = {'threshold' : '0.7', 'imgurl':'https://datahunt.s3.ap-northeast-2.amazonaws.com/uploads/37/10016369_1565766485_image1_L.jpg'}
#data = {'threshold' : '0.7'}
#resp1 = requests.post("http://localhost:5004/predict", data=data, files={"file": open(fn,'rb')})
#resp1 = requests.post("http://15.164.229.220:8080/predictions/facenet", data=data, files={"file" : open(fn, 'rb')})
resp1 = requests.post("http://13.124.166.149:8080/predictions/facenet", data=data, files={"file" : open(fn, 'rb')})
#resp1 = requests.post("http://localhost:8080/predictions/facenet", data=data, files={"file" : open(fn, 'rb')})
#resp1 = requests.post("http://localhost:8080/predictions/facenet", data=data)
#resp1 = requests.post("http://localhost:5004/predict", data=data)
#resp1 = requests.post("http://localhost:5004/predict", json=data)
#resp1 = requests.post("http://localhost:5004/predict", files={"file": open(fn,'rb')})

print("\n##########    My result    ##########")
json_dict = resp1.json()
#print("OD result(time taken: %f)" % json_dict['time_OD'])
#print(json_dict['detections'])
print(json_dict)