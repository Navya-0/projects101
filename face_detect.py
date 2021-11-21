import cv2
cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,0)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detection = cascade_classifier.detectMultiScale(gray, 1.3, 5)
    if(len(detection) > 0):
        (u,v,w,h) = detection[0]
        frame = cv2.rectangle(frame,(u,v),(u+w,v+h),(225,0,0),2)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()