import numpy as np

def get_new_eye_contour(np_landmark_a, np_landmark_b, eye_contour, eye_area):
  def get_length(np_landmark, eye_contour, right_eye_area):
    min_outside_x = np.min(np_landmark[eye_area, 0])
    max_outside_x = np.max(np_landmark[eye_area, 0])
    min_outside_y = np.min(np_landmark[eye_area, 1])
    max_outside_y = np.max(np_landmark[eye_area, 1])
    min_inside_x = np.min(np_landmark[eye_contour, 0])
    max_inside_x = np.max(np_landmark[eye_contour, 0])
    min_inside_y = np.min(np_landmark[eye_contour, 1])
    max_inside_y = np.max(np_landmark[eye_contour, 1])
    length_x = np.abs(max_outside_x - min_outside_x)
    length_y = np.abs(max_outside_y - min_outside_y)
    left_offset = np.abs(min_inside_x - min_outside_x)
    right_offset = np.abs(max_outside_x - max_inside_x)
    above_offset = np.abs(min_inside_y - min_outside_y)
    below_offset = np.abs(max_outside_y - max_inside_y)
    return length_x, length_y, left_offset, right_offset, above_offset, below_offset

  min_outside_x = np.min(np_landmark_b[eye_area, 0])
  max_outside_x = np.max(np_landmark_b[eye_area, 0])
  min_outside_y = np.min(np_landmark_b[eye_area, 1])
  max_outside_y = np.max(np_landmark_b[eye_area, 1])
  length_x_a, length_y_a, left_offset_a, right_offset_a, above_offset_a, below_offset_a =  get_length(np_landmark_a, eye_contour, eye_area)
  length_x_b, length_y_b, _, _, _, _ =  get_length(np_landmark_b, eye_contour, eye_area)
  new_left_ratio = left_offset_a/length_x_a
  new_above_ratio = above_offset_a/length_y_a
  new_right_ratio = right_offset_a/length_x_a
  new_below_ratio = below_offset_a/length_y_a
  new_min_inside_x = ((new_left_ratio)*length_x_b) + min_outside_x
  new_min_inside_y = ((new_above_ratio)*length_y_b) + min_outside_y
  new_max_inside_x =  max_outside_x - ((new_right_ratio)*length_x_b)
  new_max_inside_y =  max_outside_y - ((new_below_ratio)*length_y_b)

  x = 0; y = 1
  old_eye_contour = np_landmark_a[eye_contour, :2]
  m = (new_max_inside_x - new_min_inside_x) / (np.max(old_eye_contour[:, x]) - np.min(old_eye_contour[:, x]))
  b = new_min_inside_x - (m * np.min(old_eye_contour[:, x]))
  new_xs = (m * old_eye_contour[:, x]) + b
  new_xs = new_xs.astype(int)

  m = (new_max_inside_y - new_min_inside_y) / (np.max(old_eye_contour[:, y]) - np.min(old_eye_contour[:, y]))
  b = new_min_inside_y - (m * np.min(old_eye_contour[:, y]))
  new_ys = (m * old_eye_contour[:, y]) + b
  new_ys = new_ys.astype(int)
  return new_xs, new_ys


def adjust_mouth(ax, ay, bx, by, mouth_landmark, mouth_inside_landmark):
  # outside 
  mouth_outside_landmark = mouth_landmark.copy()
  mouth_outside_point = np.array([[ax[i], ay[i]] for i in mouth_outside_landmark])
  # inside
  # mouth_inside_landmark
  mouth_inside_point_a = np.array([[ax[i], ay[i]] for i in mouth_inside_landmark])

  mouth_inside_point_b = np.array([[bx[i], by[i]] for i in mouth_inside_landmark])
  max_inside_bx = np.max(mouth_inside_point_b[:, 0])
  max_inside_by = np.max(mouth_inside_point_b[:, 1])
  min_inside_bx = np.min(mouth_inside_point_b[:, 0])
  min_inside_by = np.min(mouth_inside_point_b[:, 1])

  # shift to new image_a mouth inside to new max min
  x = 0; y = 1
  m = (max_inside_bx - min_inside_bx) / (np.max(mouth_inside_point_a[:, x]) - np.min(mouth_inside_point_a[:, x]))
  b = min_inside_bx - (m * np.min(mouth_inside_point_a[:, x]))
  new_xs = (m * mouth_inside_point_a[:, x]) + b
  new_xs = new_xs.astype(int)

  m = (max_inside_by - min_inside_by) / (np.max(mouth_inside_point_a[:, y]) - np.min(mouth_inside_point_a[:, y]))
  b = min_inside_by - (m * np.min(mouth_inside_point_a[:, y]))
  new_ys = (m * mouth_inside_point_a[:, y]) + b
  new_ys = new_ys.astype(int)

  for i, landmark in enumerate(mouth_inside_landmark):
    bx[landmark] = new_xs[i]
    by[landmark] = new_ys[i]
  return bx, by

  