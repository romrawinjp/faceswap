import argparse
import cv2
import numpy as np

from utils import mp_detector, get_coordinate, display, update_position

# plot_landmark("..//image//another_man.jpg", annotate_number=True)

# arg parser
# parser = argparse.ArgumentParser(description='Swaping face components')
# parser.add_argument('image_a_dir', type=str)
# parser.add_argument('image_b_dir', type=str)
# parser.add_argument("--eye", default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument("--eyebrow", default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument("--nose", default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument("--mouth", default=False, action=argparse.BooleanOptionalAction)
# args = parser.parse_args()

# image_a = cv2.imread(args.image_a_dir)
# image_b = cv2.imread(args.image_b_dir)

# landmark_a = mp_detector(image_a)
# landmark_b = mp_detector(image_b)

# ax, ay, az = get_coordinate(landmark_a, image_a)
# bx, by, bz = get_coordinate(landmark_b, image_b)

def swap_component(image_a, ax, ay, image_b, bx, by, contour_area, seamless = True):
    # change template to gray
    image_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
  
    # extract triangle in image_a
    landmark_tuple_a = []
    for i in range(len(contour_area)):
        landmark_tuple_a.append((ax[contour_area[i]], ay[contour_area[i]]))

    np_landmark_a = np.array(landmark_tuple_a, np.int32)
    convexhull_a = cv2.convexHull(np_landmark_a)

    image_a_canvas = np.zeros_like(image_a_gray)
    cv2.fillConvexPoly(image_a_canvas, convexhull_a, 255)
    image_a_face = cv2.bitwise_and(image_a, image_a, mask=image_a_canvas)

    # Delaunay triangulation
    bounding_rectangle = cv2.boundingRect(convexhull_a)
    subdivisions = cv2.Subdiv2D(bounding_rectangle)
    subdivisions.insert(landmark_tuple_a)
    triangles_vector = subdivisions.getTriangleList()
    triangles_array = np.array(triangles_vector, dtype=np.int32)

    triangle_landmark_points_list = []
    # source_face_image_copy = image_a_face.copy()
    for triangle in triangles_array:
        index_point_1 = (triangle[0], triangle[1])
        index_point_2 = (triangle[2], triangle[3])
        index_point_3 = (triangle[4], triangle[5])

        index_1 = np.where((np_landmark_a == index_point_1).all(axis=1))[0][0]
        index_2 = np.where((np_landmark_a == index_point_2).all(axis=1))[0][0]
        index_3 = np.where((np_landmark_a == index_point_3).all(axis=1))[0][0]

        triangle = [index_1, index_2, index_3]
        triangle_landmark_points_list.append(triangle)
    
    # image_b
    height, width, no_of_channels = image_b.shape
    destination_image_canvas = np.zeros((height, width, no_of_channels), np.uint8)

    landmark_tuple_b = []
    for i in range(len(contour_area)):
        landmark_tuple_b.append((bx[contour_area[i]], by[contour_area[i]]))

    destination_face_landmark_points_array = np.array(landmark_tuple_b, np.int32)
    destination_face_convexhull = cv2.convexHull(destination_face_landmark_points_array)

    for i, triangle_index_points in enumerate(triangle_landmark_points_list):
        source_triangle_point_1 = landmark_tuple_a[triangle_index_points[0]]
        source_triangle_point_2 = landmark_tuple_a[triangle_index_points[1]]
        source_triangle_point_3 = landmark_tuple_a[triangle_index_points[2]]
        source_triangle = np.array([source_triangle_point_1, source_triangle_point_2, source_triangle_point_3], np.int32)

        source_rectangle = cv2.boundingRect(source_triangle)
        (x, y, w, h) = source_rectangle
        cropped_source_rectangle = image_a[y:y+h, x:x+w]

        source_triangle_points = np.array([[source_triangle_point_1[0]-x, source_triangle_point_1[1]-y], 
                                        [source_triangle_point_2[0]-x, source_triangle_point_2[1]-y], 
                                        [source_triangle_point_3[0]-x, source_triangle_point_3[1]-y]], np.int32)


        # Create a mask using cropped destination triangle's bounding rectangle(for same landmark points as used for source triangle)

        destination_triangle_point_1 = landmark_tuple_b[triangle_index_points[0]]
        destination_triangle_point_2 = landmark_tuple_b[triangle_index_points[1]]
        destination_triangle_point_3 = landmark_tuple_b[triangle_index_points[2]]
        destination_triangle = np.array([destination_triangle_point_1, destination_triangle_point_2, destination_triangle_point_3], np.int32)

        destination_rectangle = cv2.boundingRect(destination_triangle)
        (x, y, w, h) = destination_rectangle

        cropped_destination_rectangle_mask = np.zeros((h, w), np.uint8)

        destination_triangle_points = np.array(
            [
                [destination_triangle_point_1[0]-x, destination_triangle_point_1[1]-y],
                [destination_triangle_point_2[0]-x, destination_triangle_point_2[1]-y], 
                [destination_triangle_point_3[0]-x, destination_triangle_point_3[1]-y]
            ], 
            np.int32
        )

        cv2.fillConvexPoly(cropped_destination_rectangle_mask, destination_triangle_points, 255)
        
        # Warp source triangle to match shape of destination triangle and put it over destination triangle mask
        source_triangle_points = np.float32(source_triangle_points)
        destination_triangle_points = np.float32(destination_triangle_points)
        
        matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
        warped_rectangle = cv2.warpAffine(cropped_source_rectangle, matrix, (w, h))

        warped_triangle = cv2.bitwise_and(warped_rectangle, warped_rectangle, mask=cropped_destination_rectangle_mask)
        
        # Reconstructing destination face in empty canvas of destination image
        # removing white lines in triangle using masking
        new_dest_face_canvas_area = destination_image_canvas[y:y+h, x:x+w]
        new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)
        _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_created_triangle)
        new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, warped_triangle)
        destination_image_canvas[y:y+h, x:x+w] = new_dest_face_canvas_area
    
    # Put reconstructed face on the destination image
    final_destination_canvas = np.zeros_like(image_b_gray)
    final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas, destination_face_convexhull, 255)
    final_destination_canvas = cv2.bitwise_not(final_destination_face_mask)
    destination_face_masked = cv2.bitwise_and(image_b, image_b, mask=final_destination_canvas)
    destination_with_face = cv2.add(destination_face_masked, destination_image_canvas)

    if seamless:
        (x, y, w, h) = cv2.boundingRect(destination_face_convexhull)
        destination_face_center_point = (int((x+x+w+10)/2), int((y+y+h+10)/2))
        seamless_cloned_face = cv2.seamlessClone(
            destination_with_face, 
            image_b, 
            final_destination_face_mask, 
            destination_face_center_point, 
            cv2.NORMAL_CLONE)
        return seamless_cloned_face
    else: return destination_with_face

# adjustment
from faceswap.adjust import get_new_eye_contour, adjust_mouth

# replace = False
# if args.eye:
#     replace = True
#     left_eye_landmark = [7, 33, 133, 144, 145, 163, 153, 154, 155, 157, 158, 159, 160, 161, 173, 246, 468, 471, 469, 470, 472]
#     around_left_eye_landmark = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 56, 110, 112, 113, 130, 189, 190,
#                                 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 243, 244, 247]
#     right_eye_landmark = [249, 263, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 463, 466, 473, 476, 474, 475, 477]
#     around_right_eye_landmark = [252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 286, 339, 341, 342, 359, 413, 414, 441, 
#                                 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 464, 465, 467]
#     right_eye_area = right_eye_landmark + around_right_eye_landmark
#     left_eye_area = left_eye_landmark + around_left_eye_landmark
#     np_landmark_a = np.array([ax, ay, az]).T
#     np_landmark_b = np.array([bx, by, bz]).T
#     new_xs, new_ys = get_new_eye_contour(np_landmark_a, np_landmark_b, right_eye_landmark, right_eye_area)
#     bx, by = update_position(bx, by, right_eye_landmark, new_xs, new_ys)
#     new_xs, new_ys = get_new_eye_contour(np_landmark_a, np_landmark_b, left_eye_landmark, left_eye_area)
#     bx, by = update_position(bx, by, left_eye_landmark, new_xs, new_ys)
#     result = swap_component(image_a, ax, ay, image_b, bx, by, right_eye_area)
#     result = swap_component(image_a, ax, ay, result, bx, by, left_eye_area)
#     image_b = result.copy()
# if args.eyebrow:
#     replace = True
#     eyebrow_left_landmark = [53, 63, 68, 55, 65, 66, 69, 104, 105, 107, 108, 113, 225, 224, 223, 222, 221, 71, 70, 46, 124, 156, 139, 9]
#     eyebrow_right_landmark = [441, 442, 443, 444, 445, 342, 285, 336, 337, 295, 296, 299, 282, 334, 333, 283, 298, 276, 300, 301, 353, 383, 368]
#     result = swap_component(image_a, ax, ay, image_b, bx, by, eyebrow_left_landmark)
#     result = swap_component(image_a, ax, ay, result, bx, by, eyebrow_right_landmark)
#     image_b = result.copy()
# if args.nose:
#     replace = True
#     nose_landmark = [47, 193, 168, 277, 357, 371, 417, 423, 465, 391, 393, 0, 165, 167, 203, 142, 128, 245]
#     nose_inside_landmark = [1, 2, 3, 4, 5, 6, 19, 20, 44, 45, 48, 49, 51, 59, 60, 64, 75, 79, 94, 97, 98, 99, 102, 114, 115, 
#                             122, 125, 126, 129, 131, 134, 141, 164, 166, 174, 188, 195, 196, 197, 198, 209, 217, 218, 219, 220,
#                             235, 236, 237, 238, 239, 240, 241, 242, 244, 248, 250, 274, 275, 281, 289, 290, 305, 309, 326, 327,
#                             328, 344, 351, 354, 360, 363, 370, 392, 399, 412, 419, 420, 437, 438, 439, 440, 455, 456,
#                             457, 458, 459, 460, 461, 462]
#     nose_area = nose_landmark + nose_inside_landmark
#     result = swap_component(image_a, ax, ay, image_b, bx, by, nose_area)
#     image_b = result.copy()
# if args.mouth:
#     replace = True
#     distance = ((bx[13] - bx[14])**2 + (by[13] - by[14])**2)**(0.5)
#     if distance > 30:
#         mouth_landmark = [164, 165, 167, 393, 391, 92, 186, 43, 57, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322]
#         mouth_inside_landmark = [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 
#                                 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 
#                                 269, 270, 271, 272, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318,
#                                 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415]
#         mouth_inside_contour = [13, 14, 80, 81, 82, 87, 88, 178, 310, 311, 312, 317, 318, 324, 402, 415, 308, 191, 78, 95]
#         mouth_area = mouth_landmark + mouth_inside_landmark
#         image_b = result.copy()
#         original_mouth = image_b.copy()
#         result = swap_component(image_a, ax, ay, image_b, bx, by, mouth_area)
#         result = swap_component(original_mouth, bx, by, result, bx, by, mouth_inside_contour, seamless = False)
#     else:
#         mouth_landmark = [164, 165, 167, 393, 391, 92, 186, 43, 57, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322]
#         mouth_inside_landmark = [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 
#                                 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 
#                                 269, 270, 271, 272, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318,
#                                 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415]
#         mouth_area = mouth_landmark + mouth_inside_landmark
#         image_b = result.copy()
#         result = swap_component(image_a, ax, ay, image_b, bx, by, mouth_area)

# if replace != True:
#     print("You didn't select face component")
# else: 
#     from enhance import enhance_image
#     result = enhance_image(result)
#     display(result)

def swap_eyes(image_a, ax, ay, image_b, bx, by):
    left_eye_landmark = [7, 33, 133, 144, 145, 163, 153, 154, 155, 157, 158, 159, 160, 161, 173, 246, 468, 471, 469, 470, 472]
    around_left_eye_landmark = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 56, 110, 112, 113, 130, 189, 190,
                                221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 243, 244, 247]
    right_eye_landmark = [249, 263, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 463, 466, 473, 476, 474, 475, 477]
    around_right_eye_landmark = [252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 286, 339, 341, 342, 359, 413, 414, 441, 
                                442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 464, 465, 467]
    right_eye_area = right_eye_landmark + around_right_eye_landmark
    left_eye_area = left_eye_landmark + around_left_eye_landmark
    np_landmark_a = np.array([ax, ay]).T
    np_landmark_b = np.array([bx, by]).T
    new_xs, new_ys = get_new_eye_contour(np_landmark_a, np_landmark_b, right_eye_landmark, right_eye_area)
    bx, by = update_position(bx, by, right_eye_landmark, new_xs, new_ys)
    new_xs, new_ys = get_new_eye_contour(np_landmark_a, np_landmark_b, left_eye_landmark, left_eye_area)
    bx, by = update_position(bx, by, left_eye_landmark, new_xs, new_ys)
    result = swap_component(image_a, ax, ay, image_b, bx, by, right_eye_area)
    result = swap_component(image_a, ax, ay, result, bx, by, left_eye_area)
    return result

def swap_nose(image_a, ax, ay, image_b, bx, by):
    nose_landmark = [47, 193, 168, 277, 357, 371, 417, 423, 465, 391, 393, 0, 165, 167, 203, 142, 128, 245]
    nose_inside_landmark = [1, 2, 3, 4, 5, 6, 19, 20, 44, 45, 48, 49, 51, 59, 60, 64, 75, 79, 94, 97, 98, 99, 102, 114, 115, 
                            122, 125, 126, 129, 131, 134, 141, 164, 166, 174, 188, 195, 196, 197, 198, 209, 217, 218, 219, 220,
                            235, 236, 237, 238, 239, 240, 241, 242, 244, 248, 250, 274, 275, 281, 289, 290, 305, 309, 326, 327,
                            328, 344, 351, 354, 360, 363, 370, 392, 399, 412, 419, 420, 437, 438, 439, 440, 455, 456,
                            457, 458, 459, 460, 461, 462]
    nose_area = nose_landmark + nose_inside_landmark
    result = swap_component(image_a, ax, ay, image_b, bx, by, nose_area)
    return result

def swap_mouth(image_a, ax, ay, image_b, bx, by):
    mouth_landmark = [164, 165, 167, 393, 391, 92, 186, 43, 57, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322]
    mouth_inside_landmark = [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 
                            82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 
                            269, 270, 271, 272, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318,
                            319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415]
    mouth_area = mouth_landmark + mouth_inside_landmark
    distance = ((bx[13] - bx[14])**2 + (by[13] - by[14])**2)**(0.5)
    if distance > 30:
        mouth_inside_contour = [13, 14, 80, 81, 82, 87, 88, 178, 310, 311, 312, 317, 318, 324, 402, 415, 308, 191, 78, 95]
        original_mouth = image_b.copy()
        result = swap_component(image_a, ax, ay, image_b, bx, by, mouth_area)
        result = swap_component(original_mouth, bx, by, result, bx, by, mouth_inside_contour, seamless = False)
    else:
        bx, by = adjust_mouth(ax, ay, bx, by, mouth_landmark, mouth_inside_landmark)
        result = swap_component(image_a, ax, ay, image_b, bx, by, mouth_area)
    return result

def swap_eyebrows(image_a, ax, ay, image_b, bx, by):
    eyebrow_left_landmark = [53, 63, 68, 55, 65, 66, 69, 104, 105, 107, 108, 113, 225, 224, 223, 222, 221, 71, 70, 46, 124, 156, 139, 9]
    eyebrow_right_landmark = [441, 442, 443, 444, 445, 342, 285, 336, 337, 295, 296, 299, 282, 334, 333, 283, 298, 276, 300, 301, 353, 383, 368]
    result = swap_component(image_a, ax, ay, image_b, bx, by, eyebrow_left_landmark)
    result = swap_component(image_a, ax, ay, result, bx, by, eyebrow_right_landmark)
    return result

