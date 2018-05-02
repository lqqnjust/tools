import face_recognition
import cv2
from PIL import Image, ImageDraw , ImageFont
import numpy
import glob
import os
# f1 = face_recognition.load_image_file('zh/006dePAgjw1ezjam6ihttj30qo1ba125.jpg')
# f2 = face_recognition.load_image_file('zh/006dePAgjw1f8fxlf6aqcj30qo1baal0.jpg')

# face1 = face_recognition.face_encodings(f1)[0]
# face2 = face_recognition.face_encodings(f2)[0]

# face_distances = face_recognition.face_distance([face1], face2)
# print(face_distances)

encodings = []
names = []

for f in glob.glob('images/*.*'):
    print(f)
    filename = os.path.basename(f)
    name, ext = os.path.splitext(filename)
    names.append(name)
    image = face_recognition.load_image_file(f)
    face1 = face_recognition.face_encodings(image)[0]
    encodings.append(face1)


for f in glob.glob('test/*.*'):
    image = face_recognition.load_image_file(f)
    print(image.shape)

    (rows, cols,channels) = image.shape
    s_x = 400 / cols
    s_y = 400 / rows
    scale = min(s_x, s_y)

    small_image = cv2.resize(image,(0,0), fx=scale, fy=scale)
    face_locations = face_recognition.face_locations(small_image)
    face_encodings = face_recognition.face_encodings(small_image,face_locations)

    pil_image = None
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(encodings, face_encoding,tolerance=0.45)

        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]

        cv2.rectangle(small_image, (left, top), (right, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(small_image, (left, bottom ), (right, bottom+20), (0, 0, 255), cv2.FILLED)

        pil_image = Image.fromarray(small_image)
        draw = ImageDraw.Draw(pil_image)  
        font = ImageFont.truetype("C:\\WINDOWS\\Fonts\\simkai.ttf", 18)
        #draw.text((20, 120), unicode(txt, 'gbk'), font=font)  
        draw.text((left+2, bottom+4), name, font=font)  
    if not pil_image:
        pil_image = Image.fromarray(small_image)
    img = cv2.cvtColor(numpy.asarray(pil_image),cv2.COLOR_RGB2BGR) 
    cv2.imshow("dd", img)

    cv2.waitKey(1000)