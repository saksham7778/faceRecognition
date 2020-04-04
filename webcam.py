import cv2

#Init Camera
cap = cv2.VideoCapture(0)


while True:
	ret,frame = cap.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",gray_frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):   #ord is a method that tells the ascii value to character.
		break

cap.release()
cv2.destroyAllWindows()