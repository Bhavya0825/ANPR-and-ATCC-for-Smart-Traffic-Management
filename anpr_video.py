import cv2 
import numpy as np 
from skimage.filters import threshold_local 
import tensorflow as tf 
from skimage import measure 
import imutils 
import os 
import pymysql
import pytesseract
import dbconnection
import easyocr

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- HELPER FUNCTIONS ----------
def sort_cont(character_contours): 
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours] 
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                     key=lambda b: b[1][i], reverse=False))
    return character_contours 


def segment_chars(plate_img, fixed_width): 
    
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2] 
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2) 
    thresh = cv2.bitwise_not(thresh) 

    plate_img = imutils.resize(plate_img, width=fixed_width) 
    thresh = imutils.resize(thresh, width=fixed_width) 
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) 

    labels = measure.label(thresh, background=0) 
    charCandidates = np.zeros(thresh.shape, dtype='uint8') 

    characters = [] 
    for label in np.unique(labels): 
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype='uint8') 
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = cnts[1] if imutils.is_cv3() else cnts[0] 

        if len(cnts) > 0: 
            c = max(cnts, key=cv2.contourArea) 
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c) 

            aspectRatio = boxW / float(boxH) 
            solidity = cv2.contourArea(c) / float(boxW * boxH) 
            heightRatio = boxH / float(plate_img.shape[0]) 

            if aspectRatio < 1.0 and solidity > 0.15 and 0.5 < heightRatio < 0.95 and boxW > 14: 
                hull = cv2.convexHull(c) 
                cv2.drawContours(charCandidates, [hull], -1, 255, -1) 

    contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    if contours: 
        contours = sort_cont(contours) 
        addPixel = 4
        for c in contours: 
            (x, y, w, h) = cv2.boundingRect(c) 
            y = max(0, y - addPixel)
            x = max(0, x - addPixel)

            temp = bgr_thresh[y:y + h + addPixel * 2, x:x + w + addPixel * 2] 
            characters.append(temp) 
        return characters 
    
    return None


# ---------- PLATE FINDER CLASS (UNCHANGED) ----------
class PlateFinder: 
    def __init__(self, minPlateArea, maxPlateArea): 
        self.min_area = minPlateArea 
        self.max_area = maxPlateArea 
        self.element_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))

    def preprocess(self, input_img): 
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0) 
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY) 
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3) 
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, self.element_structure, morph_n_thresholded_img)
        return morph_n_thresholded_img 

    def extract_contours(self, after_preprocess): 
        contours, _ = cv2.findContours(after_preprocess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        return contours 

    def clean_plate(self, plate): 
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) 
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2) 

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        if contours: 
            areas = [cv2.contourArea(c) for c in contours] 
            max_index = np.argmax(areas) 
            max_cnt = contours[max_index] 
            x, y, w, h = cv2.boundingRect(max_cnt) 

            if not self.ratioCheck(areas[max_index], plate.shape[1], plate.shape[0]): 
                return plate, False, None
            
            return plate, True, [x, y, w, h] 
        
        return plate, False, None

    def check_plate(self, input_img, contour): 
        min_rect = cv2.minAreaRect(contour) 
        
        if self.validateRatio(min_rect): 
            x, y, w, h = cv2.boundingRect(contour) 
            after_validation = input_img[y:y + h, x:x + w] 
            cleaned, plateFound, coords = self.clean_plate(after_validation)

            if plateFound: 
                chars_on_plate = self.find_characters_on_plate(cleaned)
                if chars_on_plate is not None:     # REMOVED len==8 restriction
                    x1, y1, w1, h1 = coords 
                    coords = (x1 + x, y1 + y)
                    return cleaned, chars_on_plate, coords 
        
        return None, None, None

    def find_possible_plates(self, input_img): 
        plates = [] 
        self.char_on_plate = [] 
        self.corresponding_area = [] 

        after_preprocess = self.preprocess(input_img) 
        possible_plate_contours = self.extract_contours(after_preprocess) 

        for cnt in possible_plate_contours: 
            plate, chars_on_plate, coords = self.check_plate(input_img, cnt) 
            if plate is not None: 
                plates.append(plate) 
                self.char_on_plate.append(chars_on_plate) 
                self.corresponding_area.append(coords) 

        return plates if plates else None

    def find_characters_on_plate(self, plate): 
        return segment_chars(plate, 400)

    def ratioCheck(self, area, width, height): 
        ratio = width / float(height)
        if ratio < 1: ratio = 1 / ratio
        return self.min_area < area < self.max_area and 3 < ratio < 6

    def preRatioCheck(self, area, width, height): 
        ratio = width / float(height)
        if ratio < 1: ratio = 1 / ratio
        return self.min_area < area < self.max_area and 2.5 < ratio < 7

    def validateRatio(self, rect): 
        (x, y), (width, height), rect_angle = rect 
        angle = -rect_angle if width > height else 90 + rect_angle
        if angle > 15 or width == 0 or height == 0:
            return False
        area = width * height 
        return self.preRatioCheck(area, width, height)


# ---------- EASY OCR (Replacement only here) ----------
reader = easyocr.Reader(['en'], gpu=False)


# ===================================================================
# ========================= ANPR MAIN LOOP ==========================
# ===================================================================
def start_anpr(input_files):

    findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000)

    for file_path in input_files:

        cap = cv2.VideoCapture(file_path)

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                break

            cv2.imshow('original video', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            possible_plates = findPlate.find_possible_plates(img)

            if possible_plates:
                for i, p in enumerate(possible_plates):

                    # ---------------- EASY OCR HERE ----------------
                    result = reader.readtext(p, detail=0)
                    recognized_plate = "".join(result)

                    print("EasyOCR:", recognized_plate)

                    # Clean plate text
                    plate = recognized_plate.strip().upper()
                    plate = plate.replace(" ", "")
                    plate = ''.join([c for c in plate if c.isalnum()])

                    video_filename = os.path.basename(file_path)

                    # ---------------- DATABASE INSERT (UNCHANGED) ----------------
                    try:
                        connection = dbconnection.get_connection()
                        with connection.cursor() as cursor:

                            sql = """
                                INSERT INTO vehicle_data 
                                (number_plate, video_file, detected_at)
                                VALUES (%s, %s, NOW())
                            """
                            cursor.execute(sql, (plate, video_filename))
                            connection.commit()

                            print("DB Saved:", plate, video_filename)

                    except Exception as e:
                        print("DB Error:", e)

                    cv2.imshow('plate', p)

                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break

        cap.release()

    cv2.destroyAllWindows()


# DIRECT RUN
if __name__ == "__main__":
    video_list = [
        r"C:\Users\balas\Documents\infosys\ANPR-and-ATCC-for-Smart-Traffic-Management\sample detection videos\anpr sample.mp4"
    ]
    start_anpr(video_list)
