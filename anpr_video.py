import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
import pymysql
import dbconnection
import os

# --- Utility Function: Sort Contours (characters in order) ---
def sort_cont(character_contours):
    """To sort contours left-to-right"""
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    character_contours, _ = zip(
        *sorted(zip(character_contours, boundingBoxes), key=lambda b: b[1][0], reverse=False)
    )
    return character_contours


# --- Segment Characters from Plate ---
def segment_chars(plate_img, fixed_width):
    """Extract Value channel and segment individual characters"""
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
            y, x = max(y - addPixel, 0), max(x - addPixel, 0)
            temp = bgr_thresh[y:y + h + (addPixel * 2), x:x + w + (addPixel * 2)]
            characters.append(temp)
        return characters
    return None


# --- PlateFinder Class ---
class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph_img = threshold_img.copy()
        cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, self.element_structure, dst=morph_img)
        return morph_img

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
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None
            return plate, True, [x, y, w, h]
        return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if characters_on_plate is not None and len(characters_on_plate) >= 5:
                    x1, y1, _, _ = coordinates
                    coordinates = x1 + x, y1 + y
                    return after_clean_plate_img, characters_on_plate, coordinates
        return None, None, None

    def find_possible_plates(self, input_img):
        plates, self.char_on_plate, self.corresponding_area = [], [], []
        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)

        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)
        return plates if len(plates) > 0 else None

    def find_characters_on_plate(self, plate):
        charactersFound = segment_chars(plate, 400)
        return charactersFound if charactersFound else None

    def ratioCheck(self, area, width, height):
        min, max = self.min_area, self.max_area
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        return not ((area < min or area > max) or (ratio < 3 or ratio > 6))

    def preRatioCheck(self, area, width, height):
        min, max = self.min_area, self.max_area
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        return not ((area < min or area > max) or (ratio < 2.5 or ratio > 7))

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        angle = -rect_angle if width > height else 90 + rect_angle
        if angle > 15 or height == 0 or width == 0:
            return False
        area = width * height
        return self.preRatioCheck(area, width, height)


# --- OCR Class ---
class OCR:
    def __init__(self, modelFile, labelFile):
        self.model_file = modelFile
        self.label_file = labelFile
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto())

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        lines = tf.io.gfile.GFile(labelFile).readlines()
        for l in lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        image = cv2.resize(image, (imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, 0.5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):
        input_name, output_name = "import/input", "import/final_result"
        input_op = self.graph.get_operation_by_name(input_name)
        output_op = self.graph.get_operation_by_name(output_name)
        results = self.sess.run(output_op.outputs[0], {input_op.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            plate += self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)


# --- Main Function ---
def start_anpr(input_files):
    findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
    model = OCR(
        modelFile=r"C:\Users\balas\Documents\infosys\ANPR-and-ATCC-for-Smart-Traffic-Management\anpr model files\binary_128_0.50_ver3.pb",
        labelFile=r"C:\Users\balas\Documents\infosys\ANPR-and-ATCC-for-Smart-Traffic-Management\anpr model files\binary_128_0.50_labels_ver2.txt"
    )

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
            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)

                    print("Detected Plate:", recognized_plate)

                    connection = dbconnection.get_connection()
                    with connection.cursor() as cursor:
                        sql_query = "INSERT INTO vehicle_data (number_plate) VALUES (%s)"
                        cursor.execute(sql_query, (recognized_plate,))
                        connection.commit()
                        print("SQL Statement Executed:", sql_query)

                    cv2.imshow('plate', p)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = [r"C:\Users\balas\Documents\infosys\ANPR-and-ATCC-for-Smart-Traffic-Management\sample detection videos\anpr sample.mp4"]
    start_anpr(video_file)
