import json
from datetime import datetime

import cv2
import numpy as np
import requests 
import uuid
from pyzbar.pyzbar import decode


def qrScanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    qr_codes = decode(gray)
    for qr_code in qr_codes:
        data = qr_code.data.decode("utf-8")
        x, y, w, h = qr_code.rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, data, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    result = data.split("\n")
    return result


def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if len(approx) == 4:
            rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon

def encodeQR(imageQR):
    barcodes = decode(imageQR)
    # Lặp qua các mã QR code tìm thấy
    for barcode in barcodes:
        # Giải mã dữ liệu từ mã QR code
        qr_data = barcode.data.decode("utf-8")
        return qr_data
    

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    # print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    # print(add)
    # print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def PushData(Quizcode,userDo,QuizData):
    print("Start Join!")
    # request data from Quiz
    jsonData = {
        "practiceTestCode": Quizcode,
        "userName": userDo
    }
    url_upload = "https://admin.metalearn.vn/MobileLogin/GetTestQuiz"
    #url_upload = "http://localhost:6002/MobileLogin/GetTestQuiz"
    response_upload = requests.post(url_upload, data=jsonData, )
    if response_upload.ok:
        print("Get Data completed successfully!")
        # print(response_upload.text)

    else:
        print("Something went wrong!")

    Dict = json.loads(response_upload.text)
    #print(len(Dict["Object"]["details"]))
    # QuizCode
    count = 0

    for data in Dict["Object"]["details"]:

        userAnwser = QuizData[count]
        count = count + 1
        QuestionCode = data["Code"]
        DataQuiz = data["JsonData"]
        AllAnwser = json.loads(DataQuiz)
        CheckQuiz = 0
        for i in range(len(AllAnwser)):
            IsAnswer = AllAnwser[i]["IsAnswer"]
            if IsAnswer == True:
                CheckQuiz = CheckQuiz
                break
            else:
                CheckQuiz = CheckQuiz + 1
        CheckUser = None
        if userAnwser == "A":
            # câu 1
            CheckUser = 0
        if userAnwser == "B":
            # Câu 2
            CheckUser = 1
        if userAnwser == "C":
            #   câu 3
            CheckUser = 2
        if userAnwser == "D":
            #   câu 3
            CheckUser = 3
        if userAnwser == "E":
            # câu 1
            CheckUser = None
        if userAnwser == "N/A":
            # chưa làm
            CheckUser = None

        if CheckUser == CheckQuiz:
            Iscorrect = True
        else:
            Iscorrect = False
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        ObjectResultTestGradeOmr = {
            "Id": 1,
            "StartTime": dt_string,
            "EndTime": dt_string,
            "UserResult": CheckUser,
            "CorrectResult": CheckQuiz,
            "UserPosition": None,
            "IsCorrect": Iscorrect,
            "Device": "Mobile",
            "SessionCode": str(uuid.uuid4()),
            "NumSuggest": 0,
            "TaskCode": "",
            "QuizType": "PRACTICE",
            "QuizObjCode": Quizcode,
        }

        ListObjectResult = []
        ListObjectResult.append(ObjectResultTestGradeOmr)
        ObjectResult = json.dumps(ListObjectResult)
        Trackdiligen = {
            "ObjectType": "QUIZ",
            "ObjectCode": QuestionCode,
            "ObjectResult": ObjectResult,
            "CreatedBy": userDo,

        }
        url_upload = "https://admin.metalearn.vn/MobileLogin/TrackDilligenceOffline"
        #url_upload = "http://localhost:6002/MobileLogin/TrackDilligenceOffline"
        response_upload = requests.post(url_upload, data=Trackdiligen)
        if response_upload.ok:
            print("Upload completed successfully!")
        else:
            print("Something went wrong!")

    print("Upload completed successfully!")


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def warpPerspectiveAndExtract(rec, new_width, new_height, image):
    ansApprox = getCornerPoints(rec) # lấy 4 góc của khung chứa đáp án
    ansPointsNew = reorder(ansApprox) # sắp xếp lại
    # i3=cv2.drawContours(image.copy(), myPointsNew1, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
    ptsAns1 = np.float32(ansPointsNew)  # PREPARE POINTS FOR WARP
    ptsAns2 = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])  # PREPARE POINTS FOR WARP
    matrixAns = cv2.getPerspectiveTransform(ptsAns1, ptsAns2)  # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(image.copy(), matrixAns, (new_width, new_height))  # APPLY WARP PERSPECTIVE
    return imgWarpColored, ansPointsNew

def qr_position(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qr_codes = decode(gray)
    y = 0
    for qr_code in qr_codes:
        data = qr_code.data.decode("utf-8")
        x, y, w, h = qr_code.rect
    return y

def rotation_for_F40_T1(img):
    if img.shape[0] > img.shape[1]:
        y = qr_position(img)
        if y > int(img.shape[1]/2):
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else: 
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: 
        return img


def detect_ans(list_means, avg):
    NON_ANS = "N/A"
    ans = ""
    mean_sort = sorted(list_means, reverse= True)
    ratio = mean_sort[1]/mean_sort[0]
    if ratio > 0.93:
        return NON_ANS
    else: 
        for i, mean in enumerate(list_means):
            if mean > avg:
                if i == 0: 
                    return ans + "A"
                elif i == 1: 
                    return ans + "B"
                elif i == 2: 
                    return ans + "C"
                elif i == 3: 
                    return ans + "D"
        

def getAns(list_ans_box, index, Answer_list):
    means = list(map(np.mean, list_ans_box))
    # sort means list
    sorted_means = sorted(means)
    # get the thresh to compare
    a = sorted_means[0] / sorted_means[1]
    if a > 0.96:
        Answer_list.append("Câu " + str(index + 1) + " " + "N/A")
    else:
        if np.argmin(means) == 0:
            Answer_list.append("Câu " + str(index + 1) + " " + "A")
            pass
        if np.argmin(means) == 1:
            Answer_list.append("Câu " + str(index + 1) + " " + "B")
            pass
        if np.argmin(means) == 2:
            Answer_list.append("Câu " + str(index + 1) + " " + "C")
            pass
        if np.argmin(means) == 3:
            Answer_list.append("Câu " + str(index + 1) + " " + "D")
            pass
    list_ans_box.clear()
    return Answer_list

