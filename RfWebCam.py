import face_recognition as fr
import cv2
import numpy as np
from home import get_infor #importando arquivo home

#carregando faces e nomes
rostos_conhecidos, nomes_dos_rostos = get_infor()

#capturando tela
video_capture = cv2.VideoCapture(0)

#setando tamanho especifico para tela   
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    localizacao_faces = fr.face_locations(rgb_frame) #localiza faces
    rostos_desconhecidos = fr.face_encodings(rgb_frame, localizacao_faces) #cria uma listas com as faces presentes nos frames

    
    for (x, y, w, h), rostos_desconhecidos in zip(localizacao_faces, rostos_desconhecidos):
        resultados = fr.compare_faces(rostos_conhecidos, rostos_desconhecidos) #compara a faces conhecidas com as desconhecidas (lidas da webcam)

        face_distances = fr.face_distance(rostos_conhecidos, rostos_desconhecidos) #calcula as distancias das faces conhecidas e desconhecidas

        melhor = np.argmin(face_distances) #retorno do indice da melhor comparação

        if (resultados[melhor]):
            nome = nomes_dos_rostos[melhor] #armazenando nome
        else:
            nome = "Desconhecido"

        cv2.rectangle(frame, (h, x), (y, w), (0, 0, 255), 2)
        
        cv2.rectangle(frame, (h, w-28), (y, w), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, nome, (h + 6, w-6), font, 1.0, (255,255,255), 1)
        
        cv2.imshow('Webcam - Reconhecimento de faces com nome', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
