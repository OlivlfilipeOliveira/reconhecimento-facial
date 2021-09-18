import cv2;

xml_haar_cascade = 'haarcascade_frontalface_default.xml'

faceClassifier = cv2.CascadeClassifier(xml_haar_cascade)

capture = cv2.VideoCapture(0)

#setando tamanho especifico para tela.
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

while(not cv2.waitKey(20) & 0xFF == ord("q")):
    ret, frame_color = capture.read()


    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    #utilizando função do modelo para encontrar as faces
    faces = faceClassifier.detectMultiScale(gray)

    for x, y, w, h in faces:
        #mostrando faces
        cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow('color', frame_color)
        