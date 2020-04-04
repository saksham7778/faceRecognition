# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.

# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np 
import os
from sklearn.neighbors import KNeighborsClassifier 

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = [] 
labels = []

class_id = 0 # Labels for the given file
names = {} #Mapping btw id - name


# Data Preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#Create a mapping btw class_id and name
		names[class_id] = fx[:-4]
		# print("Loaded "+fx)
		data_item = np.load(dataset_path+fx)
		# print(data_item.shape)       # contains size as say (26,30000) it means 26 rows and 30000 columns 
		face_data.append(data_item)  #[[x*30,000] , [y*30,000] , ........]
		# print(len(face_data));

		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		# print(target.shape)          # we make a array with the same size as number of rows and afterwards 
		class_id += 1                # we will add as the 30001 column ,so to get label column
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)                  # X
face_labels  = np.concatenate(labels,axis=0)    # Y

print(face_dataset.shape)
print(face_labels.shape)

# trainset = np.concatenate((face_dataset,face_labels),axis=1)
# print(trainset.shape)                                            #dataset[[X][Y]]


# Testing 

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.1,5)
	# if(len(faces)==0):
	# 	continue

	for face in faces:
		x,y,w,h = face

		#Get the face ROI
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		hi=face_section.flatten().reshape(1,30000)
		# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',hi.shape);
		#Predicted Label (out)
		neigh = KNeighborsClassifier(n_neighbors=5)
		neigh.fit(face_dataset,face_labels )
		ans=neigh.predict(hi)
		# out = knn(trainset,face_section.flatten())
		# print("*****************",ans)
		#Display on the screen the name and rectangle around it
		pred_name = names[int(ans)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
