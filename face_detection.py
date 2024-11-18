import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    retangule, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(25, 25))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4) 

    cv2.imshow('Webcam: ', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    capture.release()
    cv2.destroyAllWindows()