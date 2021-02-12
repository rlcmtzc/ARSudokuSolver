import cv2
import numpy as np

width, height = 600, 600

def pre_process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5),1)
    img = cv2.fastNlMeansDenoising(img) 
    img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
    return img

def find_edges(img):
    edges = cv2.Canny(img, 100, 200)
    return edges

def find_contours(img):
    img_copy = img.copy()
    contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_biggest_contour(contours):
    biggest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(biggest_contour, True)
    corners = cv2.approxPolyDP(biggest_contour, 0.02*peri, True)

    corners = np.float32(corners).reshape((4,2))
    correct_corners = np.zeros((4,2))

    corners_sum = corners.sum(axis=1)
    corners_diff = np.diff(corners, axis=1)
    correct_corners[0] = corners[np.argmin(corners_sum)]
    correct_corners[1] = corners[np.argmin(corners_diff)]
    correct_corners[2] = corners[np.argmax(corners_diff)]
    correct_corners[3] = corners[np.argmax(corners_sum)]
    
    return np.float32(correct_corners)


def show_image(img, title="sudoku", wait=True):
    cv2.imshow(title, img)
    if wait and not cv2.waitKey():
        exit()

def warp_image(img, show=False):
    preprocessed_img = pre_process_image(img)
    edges = find_edges(preprocessed_img)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = find_biggest_contour(contours)
    target_corners = np.float32([[0,0], [width, 0], [0, height], [width, height]])

    H = cv2.getPerspectiveTransform(corners, target_corners)
    image_warped = cv2.warpPerspective(preprocessed_img, H, (width, height))


    if show:
        img_contour_copy = img.copy()
        img_biggest_contour_copy = img.copy()
        cv2.drawContours(img_contour_copy, contours, -1, (0, 255, 0), 3)

        for index, corner in enumerate(corners):
            cv2.circle(img_biggest_contour_copy, tuple(corner), radius=7, color=(0, 0, 255), thickness=-1)
            img_biggest_contour_copy = cv2.putText(img_biggest_contour_copy, str(index), tuple(corner), 
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                           fontScale=1, 
                                           color=(255,0,0),
                                           thickness=3) 

        show_image(preprocessed_img, title="thresholded")
        show_image(edges, title="canny")
        show_image(img_contour_copy, title="contours")
        show_image(img_biggest_contour_copy, title="biggest contour")

    return image_warped, H

def check_intersection(x,y,bbs):
    for bb in bbs:
        if bb[0] <= x and bb[1] <= y and bb[0] + bb[2] >= x and bb[1] + bb[3] >= y:
            return True, bb
    return False, None

def detect_number(number_image):
    return 1

def get_numbers(img, original, show=False):
    edges = find_edges(img)
    if show:
        bb_original = original.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detected_number_bb = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x,y,w,h = rect
        aspect_ratio = w/h
        if aspect_ratio <= 0.3 or aspect_ratio >= 0.9:
            continue
        area = w*h
        if area < 600 or area >= 2000:
            continue
        
        if show:
            cv2.rectangle(bb_original,(x,y),(x+w,y+h),(255,0,0),2)
        detected_number_bb.append(rect)
    
    detected_number_bb = set(detected_number_bb)

    grid = np.zeros((9,9))
    for row in range(1,10):
        for column in range(1, 10):
            y = int(row * 600/9) - int(600/18)
            x = int(column * 600/9) - int(600/18)

            color=(0, 0, 255)
            intersect, bb = check_intersection(x,y, detected_number_bb)
            if intersect:
                color=(0, 255, 0)
                grid[row-1, column-1] = detect_number(img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]])
               
                
            if show:
                cv2.circle(original, (x,y), radius=7, color=color, thickness=-1)
                original = cv2.putText(original, f"{row},{column}", (x,y), 
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                            fontScale=1, 
                                            color=color,
                                            thickness=3) 
    if show:
        show_image(edges, title="canny")
        show_image(bb_original, title="Detected Number BoundingBoxes")
        show_image(original, title="Detected Number Cell Positions")

    return grid

def get_grid(img, show=False):
    image_warped, H = warp_image(img, show)

    if show:
        original_warped = cv2.warpPerspective(img, H, (width, height))
    else:
        original_warped = None
    grid = get_numbers(image_warped, original_warped, show)

    return grid
        

    

    