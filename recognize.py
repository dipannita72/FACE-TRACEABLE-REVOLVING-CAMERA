import cv2
import os
import shutil
import dlib
import scipy.misc
import numpy as np
import face_recognition
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
"""
FUNCTION: tracker_func
It takes 'tracker' object and 'img' (current frame) and returns the frame with bounding box on tracking face and the tracking face info (x,y,w,h)
"""
si = 0
addtoset = 0

detector = MTCNN()   
   
def build_dataset():
    image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))
    paths_to_images = ['images/' + x for x in image_filenames]
    i = 0
    for path_to_image in paths_to_images:
        
        box = []
        img = cv2.imread(path_to_image)
        faces =  detector.detect_faces(img)
        
        for result in faces:
            x, y, w, h = result['box']
            box.append((y, x+w, y+h, x))
        if(len(faces) != 0):
            print(path_to_image)
            crop_img = img[y:y+h, x:x+w]
            encodings = face_recognition.face_encodings(crop_img, box)
            knownencodings.append(encodings[0])
            i+=1
            #print(encodings)

    
    

def get_face_encodings(path_to_image):
    # Load image using scipy
    image = cv2.imread(path_to_image)
    # Detect faces using the face detector
    detected_faces = face_detector(image, 1)
    # Get pose/landmarks of those faces
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def compare_face_encodings(known_faces, face):
       return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)


def find_match(known_faces, face):
    # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
    matches = compare_face_encodings([known_faces], face)    # Return the name of the first match
    count = 0
    for match in matches:
        if match:
            return True
        count += 1    # Return not found if no match found
    return False



"""
Main code starts from here
"""
flag = 0 # when flag = 1, it goes to 'tracking mode'
init = 0 # when init = 1, tracker object is already initialized
savedcnt = 0
image_path = r'F:\Zaima\face_recognition\test'
image_path2 = r'F:\Zaima\face_recognition\images'
# Image directory 
directory = r'F:\Zaima\face_recognition'
#create tracker
knownencodings = []

#tracker = cv2.TrackerGOTURN_create() # Goturn   
tracker = cv2.TrackerBoosting_create() # Boosting
#tracker = cv2.TrackerMIL_create() # MIL
#tracker = cv2.TrackerKCF_create() # KCF
#tracker = cv2.TrackerTLD_create() # TLD
#tracker = cv2.TrackerMedianFlow_create() # MedianFlow
#tracker = cv2.TrackerMOSSE_create() # MOSSE
#tracker = cv2.TrackerCSRT_create() # CSRT

#face_detector = dlib.get_frontal_face_detector()
#face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
#shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_encodings_in_selected = []
update_dataset = []
selected_face = ()
matches_found = 0
flag2 = 0


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
build_dataset()
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
print(len(knownencodings))

face_box = () # info (x,y,w,h) for the face to track will be here

while True: 
    # every iteration we will work with the current frame
    
    
    face_box_list = [] # info (x,y,w,h) for all the faces in the current frame will be stored in this list
    cnt = 1
    # Read the frame
    _, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb,model="hog")# detectMultiScale(img_type, scale_factor, min_neighbor)
    encodings = face_recognition.face_encodings(rgb, faces)
    
    timer = cv2.getTickCount()
    cnt = 0
    for result in encodings:
        matches = []
        matches = face_recognition.compare_faces(knownencodings,result,0.5)
        
        #print(matches)
        l = sum(matches)
        print("c "+str(l))
        print(matches)
        if l > matches_found and l >= 1/2 * len(knownencodings) :
            matches_found = l
            selected_face = cnt
            flag2 = 1
            
        
            
        # tracker_func(x,y,w,h, face_encodings_in_image[0],img)
        cnt += 1 # id increment

    if flag2 == 1:
        p1 = (faces[selected_face][3],faces[selected_face][0]) # left corner point coordinate
        p2 = (faces[selected_face][1],faces[selected_face][2]) # right corner point coordinate
        cv2.rectangle(img, p1, p2, (0,255,50), 2, 1) 
        cv2.putText(img, "Tracker Mode", (100,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,100),2);
        flag2 = 0
        matches_found = 0
    else:
        cv2.putText(img, "Try to look at the camera", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,0,255), 2)
        
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    #cv2.putText(img, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    si +=1

            
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        #build_dataset(img)
        
# Release the VideoCapture object
cap.release()
