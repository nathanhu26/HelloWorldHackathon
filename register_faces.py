
import face_recognition #face_recognition module
import cv2 #OpenCV Module
import numpy as np
import os


#Open the main webcam
video_capture = cv2.VideoCapture(0)

#List of registered names
registered = []


# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []


dir = "C:\\Users\\natha\\Desktop\\Purdue\\Projects\\Hackathon\\KnownFaces\\" #Main directory

for filename in os.listdir(dir): #set up encoding files using the images in the KnownFaces folder 

    img = face_recognition.load_image_file(os.path.join(dir + filename)) #create the image using the jpg file
    known_face_encodings.append(face_recognition.face_encodings(img)[0]) #create an encoding using the image
    known_face_names.append(filename.replace(".jpg", "")) #cut off the file extension to add the name



#Initialize 
face_locations = []
face_encodings = []
face_names = []

process_this_frame = True

while True:
    
    #Capture each frame from the current recording 
    ret, frame = video_capture.read()  
    
    #Only process every other frame
    if process_this_frame:
        #makes the frame smaller 
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) 

        # Convert the image from BGR to RGB, inverting the colors
        rgb_small_frame = small_frame[:, :, ::-1] 
    
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.4) #Higher tolerance means more leniency
            #compare_faces() Returns a list of true/false values that determine whether or not face encodings match with the already known face encodings 
                                                                                            
            name = "Unknown"

            #Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) 
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            #Register the user into the list of registered names, and write the name to a text file
            if name not in registered and name != "Unknown":
                print (registered)
                registered.append(name)
                print(name + " was registered!")

                with open(r'C:\\Users\\natha\\Desktop\\Purdue\\Projects\\Hackathon\\registered_list.txt', 'a') as f:
                    f.write(name + "\n")
                    f.close()

            face_names.append(name)
    process_this_frame = not process_this_frame

    


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color=(0, 0, 255), thickness=2) 

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) 

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Close the webcam and GUI
video_capture.release()
cv2.destroyAllWindows()

#Clear the text file with the registered names
with open(r'C:\\Users\\natha\\Desktop\\Purdue\\Projects\\Hackathon\\registered_list.txt', 'w') as f:
    f.write("")
f.close()

#########################
#WORKS CITED

#Adapted from https://github.com/ageitgey/face_recognition 9/18/2022
#OpenCV documentation: https://docs.opencv.org/4.x/ 9/18/2022
#face_recognition module documentation: https://face-recognition.readthedocs.io/ 9/18/2022

##########################
