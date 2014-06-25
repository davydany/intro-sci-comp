import numpy as np
import cv2

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

cam = cv2.VideoCapture(0)
s, img = cam.read()
windowName = "Face Detection"

cv2.namedWindow(windowName, cv2.CV_WINDOW_AUTOSIZE)
while s:
    s, img = cam.read()

    # detect body on image
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)
    cv2.imshow(windowName, img)
    
    # check for ESC key press, and stop when the user requests it 
    key = cv2.waitKey(10)
    if key == 27: 
        cv2.destroyWindow(windowName)
        break

cv2.destroyAllWindows()