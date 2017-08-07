import cv2
import numpy as np
import glob
import logging as lg
import operator

# LOGGIG
LOG_FILENAME = "TrainingPCA4-19p-BARU.log"
logHandler = lg.FileHandler(LOG_FILENAME)
logHandler.setLevel(lg.INFO)
formatter = lg.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logHandler.setFormatter(formatter)

logger = lg.getLogger(__name__)
logger.setLevel(lg.INFO)
logger.addHandler(logHandler)

logger.info("Memulai Program")

# VARIABLES
setOfImages = []
subtractAvg = []
eigenFaces = []
eigenVectors = None
eigenValues = None
covrMatx = None
weightEigenFaces = []
avg = 0
# file yang disediakan opencv untuk mendeteksi wajah
xml = 'haarcascade_frontalface_default.xml'
# Melakukan instansiasi Cascade Classifier
face_cascade = cv2.CascadeClassifier(xml)
filePathTraining = "Pic\Hasil Training"


def convertArrayToNPArray(givenArray=None):
    convertNPArray = np.asarray(givenArray)
    return convertNPArray


def getImageVector(Path=None, size=19):
    logger.info("Get Image ... ")
    print("Get Image ...")
    dimension = (size, size)
    jumlahCitra = pow(size, 2)
    index = 0
    for pathImage in glob.glob(Path + "/*.jpg"):
        if index == jumlahCitra:
            break
        index += 1
        idImage = pathImage[pathImage.rfind("/") + 1:]
        logger.info("Berkas terbaca: " + str(idImage))

        img = cv2.resize(cv2.imread(pathImage, 0), dimension)
        imgVec = img.reshape(size * size)

        setOfImages.append(imgVec)
        # setOfImagesNP = np.asarray(setOfImages)
        # logger.info("Dimensi setOfImages: " + str(setOfImages.shape))
        # logger.info("getImage Done")

        # return setOfImages



# vecOfImages=setOfImages, avg=avg
def getSubtractedAverage():
    global subtractAvg
    # subtractAvg = setOfImages.copy()
    # subtractAvg = [len(setOfImages)]
    print("subtractAvg", convertArrayToNPArray(subtractAvg).shape)
    print("setOfImages", convertArrayToNPArray(setOfImages[0]).shape)
    print("avg121212", convertArrayToNPArray(avg[0]).shape)
    print("avg121212T", cv2.transpose(avg[0]).shape)
    # res = cv2.subtract(np.float64(setOfImages[0]), avg)
    # print("res", convertArrayToNPArray(res).shape)
    # print("setOfImages[index] shape", convertArrayToNPArray(setOfImages[0]).shape)
    for index in range(0, len(setOfImages)):
        # print(index)
        # subtractAvg[index] = cv2.subtract(np.float64(setOfImages[index]), avg)
        # nm = np.subtract(np.float64(setOfImages[index]), cv2.transpose(avg))
        nm = np.subtract(np.float64(setOfImages[index]), avg[0])
        # print("nm.shape", nm.shape)
        subtractAvg.append(np.subtract(np.float64(setOfImages[index]), avg[0]))
        # return subtractAvg
        # print("subtractAvg",subtractAvg)
    # print("subtractAvg", subtractAvg)
    print("subtractAvg -lp", len(subtractAvg))


# subtractedAverage=subtractAvg
def getCovarianceMatrix():
    subtractAvgT = cv2.transpose(convertArrayToNPArray(subtractAvg))
    # print("convertArrayToNPArray(subtractAvg)", convertArrayToNPArray(subtractAvg).shape)
    # print("subtractAvgT", subtractAvgT.shape)
    covrMatx = np.matmul(subtractAvgT, subtractAvg)
    # print("covrMatx", covrMatx.shape)

    return covrMatx


def anotherPCA():
    global eigenVectors, avg
    # mean, eigenvectors2 = cv2.PCACompute(convertArrayToNPArray(setOfImages), np.mean(convertArrayToNPArray(setOfImages), axis=0).reshape(1, -1))
    avg, eigenVectors = cv2.PCACompute(convertArrayToNPArray(setOfImages),
                                       np.mean(convertArrayToNPArray(setOfImages), axis=0).reshape(1, -1))
    print("avg,", avg.shape)
    print("eigenvectors", eigenVectors.shape)
    print("eigenvectors[0]", eigenVectors[0].shape)

    # img = np.asarray(eigenVectors[0]).reshape((19, 19))
    # imgrec = cv2.resize(img, (512,512))
    # cv2.imshow("tes", imgrec)
    # cv2.waitKey(0)

def saveEigenvector():
    for index in range(0, len(eigenVectors)):
        mat = np.asanyarray(eigenVectors[index]).reshape((19,19))
        img = cv2.resize(mat, (512, 512))
        cv2.imwrite()


def getEigenFaces(kEigenface=7):
    global eigenFaces, eigenVectors
    # print("eigenVectors 232", type(eigenVectors))
    eigenFaces = eigenVectors[:kEigenface]
    print("eigenFaces", convertArrayToNPArray(eigenFaces).shape)
    print("eigenVectors", convertArrayToNPArray(eigenVectors[0]).shape)
    # return eigenFaces


def getWeightEigenFaces():
    print("Ukuran seharusnya: ", len(setOfImages), " x ", len(eigenFaces))
    omega = []
    # test = [len(setOfImages),[len(eigenFaces)]]
    print("subtractAvg-1", convertArrayToNPArray(subtractAvg).shape)
    for i in range(0, len(setOfImages)):
        for k in range(0, len(eigenFaces)):
            # test[i,[k]] = np.dot(eigenFaces[k], subtractAvg[i])
            # value = np.matmul(cv2.transpose(eigenFaces[k]), subtractAvg[i])
            # balik =cv2.transpose(convertArrayToNPArray(subtractAvg))
            # print("balik",balik.shape)
            value = np.dot(eigenFaces[k], subtractAvg[i])
            omega.append(value)
            # print("value-", k, value)
        # print("omega-", i, convertArrayToNPArray(omega).shape)
        weightEigenFaces.append(omega)
    print("weightEigenFaces", convertArrayToNPArray(weightEigenFaces).shape)
    return weightEigenFaces


def TrainingPCA(path="Pic/", size=19, kEigenFace=7):
    getImageVector(Path=path, size=size)
    anotherPCA()
    getSubtractedAverage()
    print("eigenVectors232", type(eigenVectors))
    getEigenFaces(kEigenFace)
    weightEigenFaces = getWeightEigenFaces()

    return weightEigenFaces


def ujiPCA():
    pass


def webcamHandler():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened:
        print("Webcam tidak berjalan")
        exit()

    while True:
        retval, frame = cap.read()

        if retval == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in face:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                crop = gray[y:y + h, x:x + w]
                resize = cv2.resize(crop, (19, 19))
                reshapedImage = resize.reshape(19 * 19)
                # print("reshapedImage", reshapedImage.shape)
                # print(" avg12",type(avg[0][0]))
                unkFaces = cv2.subtract(np.float32(reshapedImage), avg[0])
                unkOmega = []
                for k in range(0, len(eigenFaces)):
                    # test[i,[k]] = np.dot(eigenFaces[k], subtractAvg[i])
                    # value = np.matmul(cv2.transpose(eigenFaces[k]), subtractAvg[i])
                    # balik =cv2.transpose(convertArrayToNPArray(subtractAvg))
                    # print("balik",balik.shape)
                    value = np.dot(eigenFaces[k], unkFaces)
                    unkOmega.append(value)
                    # Hitung Jarak
                jarak = {}
                jarak2 = {}
                for index in range(0, len(unkOmega)):
                    jarak[index] = np.absolute(unkOmega[index] - weightEigenFaces[index])
                    jarak2[index] = np.linalg.norm(jarak[index])
                    # print("jarak: ",index," ", jarak2[index])
                # print("jarak2",jarak2)
                # jarak2 = sorted([(v, k) for (k, v), in jarak2.items()])
                jarak2 = sorted(jarak2.items(), key=operator.itemgetter(1))

                print("Citra nomor: ", jarak2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "result): " + str(jarak2), (x + w, y + h), font, 0.5, (0, 247, 255), 2,
                            cv2.LINE_AA)
        cv2.imshow('Hasil capture', frame)
            # Membuat tombol esc untuk keluar dari program
        key = cv2.waitKey(1)
        if key == 27:  # tombol keluar
            break
        elif key == ord('x'):
            print("Salah")
    cap.release()
    cv2.destroyAllWindows()




        # result = 0


wg = TrainingPCA(path="Pic\Training", size=19, kEigenFace=7)
webcamHandler()
# print("WG: ", wg)

