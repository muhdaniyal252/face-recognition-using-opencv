import cv2
import face_recognition
import os
import numpy as np

path = 'images'

images = []
names = []
image_list = os.listdir(path)

for image_name in image_list:
    img = cv2.imread(f'{path}/{image_name}')
    images.append(img)
    names.append(os.path.splitext(image_name)[0])

def find_encodings(images):
    encoded_images = []
    for img in images:
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_image = face_recognition.face_encodings(img)[0]
        encoded_images.append(encoded_image)
    return encoded_images

encoded_images = find_encodings(images)
print(len(encoded_images))

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    # img = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(img)
    encoded_faces = face_recognition.face_encodings(img,faces)

    for encoded_face,face in zip(encoded_faces,faces):
        matches = face_recognition.compare_faces(encoded_images,encoded_face)
        face_distance = face_recognition.face_distance(encoded_images,encoded_face)
    
        index_match = np.argmin(face_distance)


        if matches[index_match] and min(face_distance) < 0.55:
            print(face_distance)
            name = names[index_match]
            y1,x2,y2,x1 = face
            cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow("test", img)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cap.release()

cv2.destroyAllWindows()