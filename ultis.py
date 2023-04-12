import cv2

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance * 2.54  # centimeters


def face_data(image):
    face_width = 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        face_width = w
        # Drwaing circle at the center of the face
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y

    return face_width, faces, face_center_x, face_center_y