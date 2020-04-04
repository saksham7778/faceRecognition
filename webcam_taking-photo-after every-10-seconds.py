import cv2

#Init Camera
cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
count=0
dataset_path = './images/'

while True:
	ret,frame = cap.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if count%(10*fps) == 0 :
         cv2.imwrite('frame%d.png'%count,frame)
         cv2.imwrite('grayScale-frame%d.png'%count,gray_frame)
         print('successfully written 10th frame')
    
    

	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",gray_frame)
	count+=1
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):   #ord is a method that tells the ascii value to character.
		break

cap.release()
cv2.destroyAllWindows()