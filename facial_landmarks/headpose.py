from imutils import face_utils
import dlib
import cv2
import numpy as np 
import os
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = os.path.join(os.path.join(os.getcwd(),'facial_landmarks'),"shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(p)
k = 1
def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[2], css[1], css[0], css[3])

#3D image points
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
camera_matrix = np.zeros((3,3))
def set_camera_matrix(frame_height, frame_width):
    global camera_matrix
    center = (frame_width//2, frame_height//2)
    camera_matrix = np.array(
                            [[frame_width, 0, center[0]],
                            [0, frame_width, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )      

def head_pose_estimation(image, rects): 
    rects = [_css_to_rect(x) for x in rects]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #2D image points
        image_points = np.array([ shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        for p in image_points:
            cv2.circle(image, (int(p[0]*k), int(p[1]*k)), 3, (0,0,255), -1)
        p1 = ( int(image_points[0][0]*k), int(image_points[0][1]*k))
        p2 = ( int(nose_end_point2D[0][0][0]*k), int(nose_end_point2D[0][0][1]*k))
        cv2.line(image, p1, p2, (255,0,0), 2)
    return image
