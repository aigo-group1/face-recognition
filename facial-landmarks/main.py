from imutils import face_utils
import dlib
import cv2
import numpy as np 
import os
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = os.path.join(os.path.join(os.getcwd(),'facial-landmarks'),"shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#3D image points
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
                        
cap = cv2.VideoCapture('nowone.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Camera internals
center = (frame_width/2, frame_height/2)
camera_matrix = np.array(
                        [[frame_width, 0, center[0]],
                        [0, frame_width, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )           
# Define the codec and create VideoWriter object.The output is stored in 'output.mp4' file.
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width,frame_height))
while True:
    # load the input image, resize half and convert it to grayscale
    _, im = cap.read()
    image = np.copy(im)
    size = image.shape
    image = cv2.resize(image,(size[1]//2, size[0]//2))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        cv2.rectangle(im,(rect.left()*2, rect.top()*2),(rect.right()*2, rect.bottom()*2), (0,255,0), 2)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #2D image points
        image_points = np.array([ shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        for p in image_points:
            cv2.circle(im, (int(p[0]*2), int(p[1]*2)), 3, (0,0,255), -1)
        p1 = ( int(image_points[0][0]*2), int(image_points[0][1]*2))
        p2 = ( int(nose_end_point2D[0][0][0]*2), int(nose_end_point2D[0][0][1]*2))
        cv2.line(im, p1, p2, (255,0,0), 2)
    # show the output image with the face detections + facial landmarks
    cv2.imshow('frame', im)
    #out.write(im)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
