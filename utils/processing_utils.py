import cv2
import numpy as np
from .number_classifier import NumberClassifier

class PreProcessor():
    def __init__(self, classifier, width, height):
        self.width = width
        self.height = height
        self.classifier = classifier


        self._grid_coordinates = np.zeros((9,9,2))
        self._H = None
        self._H_inv = None
        self._number_scales = np.ones((9,9))
    
    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        if not isinstance(classifier, NumberClassifier):
            raise TypeError("PreProcessor: classifier must be of class NumberClassifier")
        self._classifier = classifier

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        if not isinstance(width, int):
            raise TypeError("PreProcessor: width must be of type int")
        if width <= 0:
            raise ValueError("PreProcessor: width cannot be smaller or equal to 0")
        else:
            self._width = width
    
    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        if not isinstance(height, int):
            raise TypeError("PreProcessor: height must be of type int")
        if height <= 0:
            raise ValueError("PreProcessor: height cannot be smaller or equal to 0")
        else:
            self._height = height

    def pre_process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5,5),1)
        img = cv2.fastNlMeansDenoising(img) 
        img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
        return img

    def find_edges(self, img):
        edges = cv2.Canny(img, 100, 200)
        return edges

    def find_contours(self, img):
        img_copy = img.copy()
        contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_biggest_contour(self, contours):
        biggest_contour = max(contours, key=cv2.contourArea)
        
        peri = cv2.arcLength(biggest_contour, True)
        corners = cv2.approxPolyDP(biggest_contour, 0.02*peri, True)
        height, _, width = corners.shape

        corners = np.float32(corners).reshape((height,width))
        correct_corners = np.zeros((4,2))

        corners_sum = corners.sum(axis=1)
        corners_diff = np.diff(corners, axis=1)
        correct_corners[0] = corners[np.argmin(corners_sum)]
        correct_corners[1] = corners[np.argmin(corners_diff)]
        correct_corners[2] = corners[np.argmax(corners_diff)]
        correct_corners[3] = corners[np.argmax(corners_sum)]
        
        return np.float32(correct_corners)


    def show_image(self, img, title="sudoku", wait=True):
        cv2.imshow(title, img)
        if wait and not cv2.waitKey():
            exit()

    def warp_image(self, img, show=False):
        preprocessed_img = self.pre_process_image(img)
        edges = self.find_edges(preprocessed_img)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # img_contour_copy = img.copy()
        # cv2.drawContours(img_contour_copy, contours, -1, (0, 255, 0), 3)
        # self.show_image(img_contour_copy, title="thresholded")
        corners = self.find_biggest_contour(contours)

        target_corners = np.float32([[0,0], [self.width, 0], [0, self.height], [self.width, self.height]])

        start_width_x = (corners[1,0] - corners[0,0])
        stop_width_x = (corners[3,0] - corners[2,0])

        start_width_y = (corners[2,1] - corners[0,1])
        stop_width_y = (corners[3,1] - corners[1,1])
        col_dist = np.linspace(start_width_x, stop_width_x, num=9)
        row_dist = np.linspace(start_width_y, stop_width_y, num=9)

        self._number_scales = np.sqrt(np.outer(col_dist, row_dist))/(81)

        self._H = cv2.getPerspectiveTransform(corners, target_corners)
        self._H_inv = cv2.getPerspectiveTransform(target_corners, corners)
        image_warped = cv2.warpPerspective(preprocessed_img, self._H, (self.width, self.height))
        

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

            self.show_image(preprocessed_img, title="thresholded")
            self.show_image(edges, title="canny")
            self.show_image(img_contour_copy, title="contours")
            self.show_image(img_biggest_contour_copy, title="biggest contour")

        return image_warped

    def check_intersection(self, x,y,bbs):
        for bb in bbs:
            if bb[0] <= x and bb[1] <= y and bb[0] + bb[2] >= x and bb[1] + bb[3] >= y:
                return True, bb
        return False, None

    def get_numbers(self, img, original, show=False):
        edges = self.find_edges(img)
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

        grid = np.zeros((9,9), dtype=int)
        for row in range(1,10):
            for column in range(1, 10):
                x_offset = int(column * 600/9)
                y_offset = int(row * 600/9)
                x = x_offset - int(600/18)
                y = y_offset - int(600/18)

                color=(0, 0, 255)
                intersect, bb = self.check_intersection(x,y, detected_number_bb)

                self._grid_coordinates[row - 1, column - 1, 0] = x_offset - int(600/9)
                self._grid_coordinates[row - 1, column - 1, 1] = y
                if intersect:
                    color=(0, 255, 0)
                    grid[row-1, column-1] = self.classifier.predict(img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]])
                    
                if show:
                    cv2.circle(original, (x,y), radius=7, color=color, thickness=-1)
                    original = cv2.putText(original, f"{row},{column}", (x,y), 
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                                fontScale=1, 
                                                color=color,
                                                thickness=3) 

        if show:
            self.show_image(edges, title="canny")
            self.show_image(bb_original, title="Detected Number BoundingBoxes")
            self.show_image(original, title="Detected Number Cell Positions")

        return grid

    def get_grid(self, img, show=False):
        image_warped = self.warp_image(img, show)
        if image_warped is None:
            return
            
        if show:
            original_warped = cv2.warpPerspective(img, self._H, (self.width, self.height))
        else:
            original_warped = None
        grid = self.get_numbers(image_warped, original_warped, show)

        return grid

    def draw_grid(self, grid, img,  x_offset=10, y_offset=10, without_detection=False, font_scale_divisor=3.5, show=False):
        
        if without_detection:
            self.warp_image(img)
            
        image_grid = np.zeros((self.width, self.height, 3))
        
        for row in range(9):
            for column in range(9):
                if grid[row, column] == 0:
                    continue
                x = int(self._grid_coordinates[row, column, 0]) + x_offset
                y = int(self._grid_coordinates[row, column, 1]) + y_offset
                image_grid = cv2.putText(image_grid, str(grid[row, column]), (x,y), 
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                        fontScale=self._number_scales[row, column]/font_scale_divisor, 
                                        color=(237, 81, 66),
                                        thickness=3) 
        if show:
            self.show_image(image_grid)

        height, width, _ = img.shape
        image_grid_warped = cv2.warpPerspective(image_grid, self._H_inv, (width, height ))
        if show:
            self.show_image(image_grid_warped)

        final_image = cv2.addWeighted(np.array(img, dtype=np.float32),0.7,np.array(image_grid_warped, dtype=np.float32), 0.7, 0)

        final_image = cv2.convertScaleAbs(final_image)
        final_image = cv2.normalize(final_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

        return final_image
        
        

    

    