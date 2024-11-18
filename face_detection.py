import cv2

# Classificador de rosto pré-treinado
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Serve para usar a câmera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    retangule, frame = capture.read()
    
    if not retangule:
        print("Erro ao capturar frame da webcam.")
        break
    
    # Converte em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Detectar objetos (rostos)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(25, 25))

    # Serve para definir a posição do rosto e adicionar o retângulo verde no rosto
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4) 

    cv2.imshow('Webcam: ', frame)

    # Pressione a tecla "q" para fechar o programa
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera a câmera do sistema
capture.release()

# Fecha as janelas abertas pelo opencv
cv2.destroyAllWindows()