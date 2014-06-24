import cv2
cam = cv2.VideoCapture(0)
s, img = cam.read()
windowName = "Face Detection"
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

cv2.namedWindow(windowName, cv2.CV_WINDOW_AUTOSIZE)

while s:
  cv2.imshow(windowName ,img )
  s, img = cam.read()

  ## detect face on image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
      for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

  # check for ESC key press, and stop when the user requests it 
  key = cv2.waitKey(10)
  if key == 27: 
    cv2.destroyWindow(windowName)
    break