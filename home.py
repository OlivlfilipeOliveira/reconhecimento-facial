import face_recognition as fr #Import face recognition
import cv2 #import do openCV
import numpy as np #import Numpy

#carregando fotos e extraindo faces
def reconhece_face(url_foto):
  foto = fr.load_image_file(url_foto)
  rostos = fr.face_encodings(foto)

  if(len(rostos)>0):
    return True, rostos
  else:
    return False, []

#carregando fotos para comparação
def get_infor():
    rostos_conhecidos = []
    nomes_dos_rostos = []
    for i in range(1, 6):
        filipe = reconhece_face('imagens/Filipe'+str(i)+'.jpeg')
        rostos_conhecidos.append(filipe[1][0])
        nomes_dos_rostos.append('Filipe')

        renata = reconhece_face('imagens/Renata'+str(i)+'.jpeg')
        rostos_conhecidos.append(renata[1][0])
        nomes_dos_rostos.append('Renata')

    return rostos_conhecidos, nomes_dos_rostos


        