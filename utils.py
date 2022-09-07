import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

import sys
sys.path.insert(0, '..')

def display(image, image_dir=None, size=(5, 5)):
    fig = plt.figure(figsize=size)
    if image_dir is not None:
        image = cv2.imread(image_dir)
    plt.imshow(image[:, :, ::-1])
    plt.axis("off")
    plt.show()

def mp_detector(image):
    # define mediapipe landmark detactor
    mp_face_mesh = mp.solutions.face_mesh
    detector = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    )
    # detect face lanmark
    results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results
    

def plot_landmark(image_dir, annotate_number = False):
    image = cv2.imread(image_dir)
    
    # mediapipe landmark detactor
    results = mp_detector(image)

    annotated_image = image.copy()
    face_landmark = results.multi_face_landmarks
    width, height = image.shape[:2]
    x = []
    y = []
    z = []
    for i in range(len(face_landmark[0].landmark)):
        xs = int((face_landmark[0].landmark[i].x)*width)
        ys = int((face_landmark[0].landmark[i].y)*height)
        zs = face_landmark[0].landmark[i].z
        x.append(xs)
        y.append(ys)
        z.append(zs)
        annotated_image = cv2.circle(
            annotated_image, 
            (xs, ys), 
            radius = 2, 
            color = (0, 0, 255), 
            thickness = 4
        )
        if annotate_number:
            annotated_image = cv2.putText(
                annotated_image, 
                text = str(i), 
                org = (xs, ys), 
                fontScale = 1, 
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                color = (255, 255, 255),
                thickness = 1
            )
    display(annotated_image)

def get_coordinate(landmark, image):
    face_landmark = landmark.multi_face_landmarks
    x = []
    y = []
    z = []
    width, height = image.shape[:2]
    for i in range(len(face_landmark[0].landmark)):
        xs = int((face_landmark[0].landmark[i].x)*width)
        ys = int((face_landmark[0].landmark[i].y)*height)
        zs = face_landmark[0].landmark[i].z
        x.append(xs)
        y.append(ys)
        z.append(zs)
    return x, y, z
# image_dir = ".//image//famale_template.png"
# # display(image_dir)
# plot_landmark(image_dir)

def update_position(bx, by, landmark, new_xs, new_ys):
    for i, landmark in enumerate(landmark):
        bx[landmark] = new_xs[i]
        by[landmark] = new_ys[i]
    return bx, by

def find_point_inside(ax, ay, contour_landmark, face_landmark_tuple):
    xs = []
    ys = []
    contour = []
    # find the point inside the contour
    for j in contour_landmark:
        xs.append(ax[j])
        ys.append(ay[j])
        contour.append((ax[j], ay[j]))

    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    polygon = Polygon(contour)

    landmark_inside = []
    for i in range(len(ax)):
        if polygon.contains(Point(ax[i], ay[i])):
            landmark_inside.append(face_landmark_tuple.index((ax[i], ay[i])))
    return landmark_inside