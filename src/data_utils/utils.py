# src/data_utils/utils.py
import cv2
import numpy as np

def resize_and_pad(image, target_size=(640, 640)):
    """
    Resizes an image to a target size while maintaining aspect ratio by padding.

    Args:
        image (np.ndarray): Input image (OpenCV format: H, W, C).
        target_size (tuple): Desired output size (width, height).

    Returns:
        tuple: Contains:
            - np.ndarray: The resized and padded image.
            - float: The resize ratio used.
            - tuple: The padding added (dw, dh) - padding added to each side (left/right, top/bottom).
    """
    h_orig, w_orig = image.shape[:2]
    w_target, h_target = target_size

    # Calculate resize ratio
    ratio = min(w_target / w_orig, h_target / h_orig)
    w_new, h_new = int(w_orig * ratio), int(h_orig * ratio)

    # Resize the image
    image_resized = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    dw = (w_target - w_new) / 2
    dh = (h_target - h_new) / 2

    # Determine top, bottom, left, right padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Create padded image (using gray color padding)
    color = [114, 114, 114] # Gray padding, common practice
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image_padded, ratio, (dw, dh)


def scale_coordinates(coords, ratio, padding):
    """
    Scales bounding box coordinates based on resize ratio and padding.

    Args:
        coords (tuple): Original coordinates (xmin, ymin, xmax, ymax).
        ratio (float): The resize ratio used by resize_and_pad.
        padding (tuple): The padding (dw, dh) added by resize_and_pad.

    Returns:
        tuple: Scaled coordinates (xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled)
               relative to the padded image size.
    """
    xmin, ymin, xmax, ymax = coords
    dw, dh = padding

    xmin_scaled = xmin * ratio + dw
    ymin_scaled = ymin * ratio + dh
    xmax_scaled = xmax * ratio + dw
    ymax_scaled = ymax * ratio + dh

    return xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled

def scale_polygon_coordinates(poly_coords, ratio, padding):
    """
    Scales polygon coordinates based on resize ratio and padding.

    Args:
        poly_coords (list): List of [x, y] original coordinates.
        ratio (float): The resize ratio used by resize_and_pad.
        padding (tuple): The padding (dw, dh) added by resize_and_pad.

    Returns:
        list: List of [x_scaled, y_scaled] coordinates relative to the padded image size.
    """
    dw, dh = padding
    scaled_poly = []
    for x, y in poly_coords:
        x_scaled = x * ratio + dw
        y_scaled = y * ratio + dh
        scaled_poly.append([x_scaled, y_scaled])
    return scaled_poly

def polygon_to_bbox(scaled_poly_coords):
    """
    Calculates the axis-aligned bounding box (xmin, ymin, xmax, ymax)
    from scaled polygon coordinates.

    Args:
        scaled_poly_coords (list): List of [x_scaled, y_scaled] coordinates.

    Returns:
        tuple: (xmin, ymin, xmax, ymax) or None if input is empty.
    """
    if not scaled_poly_coords:
        return None
    np_points = np.array(scaled_poly_coords)
    xmin = np.min(np_points[:, 0])
    ymin = np.min(np_points[:, 1])
    xmax = np.max(np_points[:, 0])
    ymax = np.max(np_points[:, 1])
    return xmin, ymin, xmax, ymax

def convert_to_yolo_format(box, img_size, class_id=0):
    """
    Converts absolute bbox coordinates (xmin, ymin, xmax, ymax) to
    normalized YOLO format (class_id, x_center, y_center, width, height).

    Args:
        box (tuple): (xmin, ymin, xmax, ymax) in absolute pixel coords.
        img_size (tuple): Target image size (width, height).
        class_id (int): The class index for the object (default 0 for license_plate).

    Returns:
        str: YOLO format string, or None if coords are invalid.
    """
    xmin, ymin, xmax, ymax = box
    img_w, img_h = img_size

    # Clamp coordinates to image boundaries to prevent invalid values
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    # Calculate width and height
    box_w = xmax - xmin
    box_h = ymax - ymin

    if box_w <= 0 or box_h <= 0:
        print(f"Warning: Skipping invalid box with zero or negative dimensions: {box}")
        return None

    # Calculate center coordinates
    x_center = xmin + box_w / 2
    y_center = ymin + box_h / 2

    # Normalize
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    width_norm = box_w / img_w
    height_norm = box_h / img_h

    # Clamp normalized values to [0.0, 1.0]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))


    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"