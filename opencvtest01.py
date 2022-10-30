import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def image_demo():
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg")

    cv.imshow("input", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def color_space_demo():
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    cv.imshow("gray", gray)
    cv.imshow("hsv", hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()


def mat_demo():
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg")
    print(image)
    h, w, c = image.shape
    roi = image[120:220, 110:185, :]

    # 创建一个图片
    blank = np.zeros((h, w, c), dtype=np.uint8)
    # blank[120:220, 110:185, :] = image[120:220, 110:185, :]
    # blank=np.copy(image)
    blank = image
    cv.imshow("blank", blank)
    # cv.imshow("image", image)
    cv.imshow("roi", roi)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 颜色取反
def pixel_demo():
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg")
    cv.imshow("input", image)
    h, w, c = image.shape

    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            image[row, col] = (255 - b, 255 - g, 255 - r)
    cv.imshow("result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def video_demo():  # 视频读取和保存
    cap = cv.VideoCapture("C:\\Users\\joysu\\Videos\\inputOpencv\\dancing.mp4")
    fourcc = cap.get(cv.CAP_PROP_FOURCC)
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    out = cv.VideoWriter("C:\\Users\\joysu\\Videos\\outputOpencv\\dancing.mp4",
                         cv.CAP_ANY, np.int(fourcc), fps, (np.int(w), np.int(h)), True)
    print(w, h, fps, fourcc)
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        cv.imshow("frame", frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        cv.imshow("result", hsv)
        out.write(hsv)
        c = cv.waitKey(10)
        if c == 27:
            break

    cv.destroyAllWindows()


def hist_demo():  # 直方图
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg")
    cv.imshow("input", image)
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        print(hist)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


def hist2d_demo():  # 二维直方图
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg")
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [48, 48], [0, 180, 0, 256])
    dst = cv.resize(hist, (400, 400))
    cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)
    cv.imshow("input", image)
    dst = cv.applyColorMap(np.uint8(dst), cv.COLORMAP_JET)
    cv.imshow("hist", dst)
    plt.imshow(hist, interpolation='nearest')
    plt.title("2D Histogram")

    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


def eqhist_demo():  # 直方图均衡化，是图片的对比度增强
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg", cv.IMREAD_GRAYSCALE)
    cv.imshow("input", image)
    result = cv.equalizeHist(image)

    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def blur_conv_demo():  # 图片卷积操作，可使图片模糊化（均值模糊）
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg", cv.IMREAD_GRAYSCALE)
    cv.imshow("input", image)
    result = cv.blur(image, (5, 5))

    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def gauss_conv_demo():  # 图片卷积操作，可使图片模糊化（高斯模糊）
    image = cv.imread("C:\\Users\\joysu\\Pictures\\fireimage\\2.jpg", cv.IMREAD_GRAYSCALE)
    cv.imshow("input", image)
    result = cv.GaussianBlur(image, (5, 5), 15)

    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def bifilter_demo():  # 图片卷积操作，可使图片模糊化（高斯双边模糊 有轮廓的地方不模糊）
    image = cv.imread("C:\\Users\\joysu\\Pictures\\joysun\\2015.jpg")
    cv.imshow("input", image)
    result = cv.bilateralFilter(image, 0, 100, 10)

    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


model_bin = "C:\\Users\\joysu\\PycharmProjects\\pythonProject01\\opencv_face_detector_uint8.pb"
config_text = "C:\\Users\\joysu\\PycharmProjects\\pythonProject01\\opencv_face_detector.pbtxt"


def face_detection_demo():  # 视频读取和保存
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    cap = cv.VideoCapture("C:\\Users\\joysu\\Videos\\hz.mp4")

    while True:
        ret, frame = cap.read()
        w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        if ret is not True:
            break
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        outs = net.forward()
        for detection in outs[0, 0, :, :]:
            score = float(detection[2])
            print(score)
            if score > 0.5:
                left = detection[3] * w
                top = detection[4] * h
                right = detection[5] * w
                bottom = detection[6] * h
                cv.rectangle(frame, (np.int(left), np.int(top)), (np.int(right), np.int(bottom)), (0, 0, 255), 2, 8, 0)

        cv.imshow("frame", frame)

        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    # image_demo()
    # color_space_demo()
    # mat_demo()
    # pixel_demo()
    video_demo()
    # hist_demo()
    # hist2d_demo()
    # eqhist_demo()
    # blur_conv_demo()
    # gauss_conv_demo()
    # bifilter_demo()
    # face_detection_demo()
