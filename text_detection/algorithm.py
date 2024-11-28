import cv2
import numpy as np

model_path = "models/"

def rect_to_points(rect):
    x, y, w, h = rect
    return np.array(((x, y), (x+w, y), (x+w, y+h), (x, y+h)))

def rects_to_polys(xs):
    return list(rect_to_points(x) for x in xs)

class Text_SWT:
    def __init__(self):
        pass

    def process(self, img):
        rects, draw, chainBBs = cv2.text.detectTextSWT(img, True)
        return (rects_to_polys(rects), rects_to_polys(chainBBs[0]), True)

class DNN_DB:
    db_model_path = model_path + "DB_TD500_resnet50.onnx"
    # db_model_path = model_path + "DB_finetune_COO.onnx"
    def __init__(self):
        self.model = cv2.dnn.TextDetectionModel_DB(self.db_model_path)
        self.model.setBinaryThreshold(0.25).setPolygonThreshold(0.3).setMaxCandidates(200).setUnclipRatio(2.0)
        scale = 1. / 255.0
        mean = (122.67891434, 116.66876762, 104.00698793)
        # model.setPolygonThreshold(0.8)
        # model.setBinaryThreshold(0.5)
        # mean = (30, 0, 30)
        size = (736, 736)
        self.model.setInputParams(scale, size, mean, False, False)

    def process(self, img):
        results, _confidences = self.model.detect(img)
        # print(_confidences)
        return (results, results, False)

class DNN_EAST:
    east_model_path = model_path + "frozen_east_text_detection.pb"
    def __init__(self):
        self.model = cv2.dnn.TextDetectionModel_EAST(self.east_model_path)
        confThreshold = 0.5
        nmsThreshold = 0.4
        self.model.setConfidenceThreshold(confThreshold).setNMSThreshold(nmsThreshold)

        detScale = 1.0
        detInputSize = (320, 320)
        detMean = (123.68, 116.78, 103.94)
        swapRB = False
        self.model.setInputParams(detScale, detInputSize, detMean, swapRB)

    def process(self, img):
        results, confidences = self.model.detect(img)
        # print(confidences)
        return (results, results, False)

class Text_CNNTextBoxes:
    def __init__(self):
        self.detector = cv2.text.TextDetectorCNN_create(model_path + "textbox.prototxt", model_path + "TextBoxes_icdar13.caffemodel")
        self.thres = 0.015
    def process(self, img):
        rects, outProbs = self.detector.detect(img)
        confident_rects = rects[np.array(outProbs) > self.thres]
        polys = rects_to_polys(confident_rects)
        return (polys, polys, False)

class Text_ER:
    def __init__(self):
        self.erc1 = cv2.text.loadClassifierNM1(model_path + 'trained_classifierNM1.xml')
        # cv::text::createERFilterNM1 (const Ptr< ERFilter::Callback > &cb, int thresholdDelta=1, float minArea=(float) 0.00025, float maxArea=(float) 0.13, float minProbability=(float) 0.4, bool nonMaxSuppression=true, float minProbabilityDiff=(float) 0.1)
        self.er1 = cv2.text.createERFilterNM1(self.erc1, 16, 0.00015, 0.5, 0.2, True, 0.1)
        
        self.erc2 = cv2.text.loadClassifierNM2(model_path + 'trained_classifierNM2.xml')
        self.er2 = cv2.text.createERFilterNM2(self.erc2, 0.8)

    def process(self, img):
        channels = list(cv2.text.computeNMChannels(img))
        # Append negative channels to detect ER- (bright regions over dark background)
        # cn = len(channels)-1
        # for c in range(0,cn):
        #     channels.append(255-channels[c])
        # Greyscale
        # channels.append(cv2.imread(img_path,0))
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # v = np.median(gray)
        # sigma = 0.33
        # low = int(max(0, (1.0 - sigma) * v))
        # high = int(max(255, (1.0 + sigma) * v))
        # canny = cv2.Canny(gray, low, high)
        # channels.append(canny)

        # Apply the default cascade classifier to each independent channel (could be done in parallel)
        rects = []
        chains = []
        for channel in channels:
            regions = cv2.text.detectRegions(channel, self.er1, self.er2)
            if len(regions) == 0:
                continue
            # rects.extend([cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions])
            # Use minAreaRect for more fine-grained coverage
            # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
            rects.extend([cv2.boxPoints(cv2.minAreaRect(p.reshape(-1, 1, 2))).astype(np.int32) for p in regions])
            chains.extend(cv2.text.erGrouping(img, channel, [r.tolist() for r in regions]))
            # rects = cv2.text.erGrouping(line,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,model_path + 'trained_classifier_erGrouping.xml',0.5)
        return (rects, rects_to_polys(chains), True)

class Stub:
    def __init__(self):
        pass
    def process(self, img):
        return ([], [], False)

def enable_CUDA_if_available(method):
    try:
        method.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        method.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except AttributeError:
        print("Method does not support using CUDA")
