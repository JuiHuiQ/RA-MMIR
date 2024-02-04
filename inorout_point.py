import cv2
import numpy as np

# path = r"L:\\dataset\\VOC2012_\\VOC2012_att\\SegmentationClassAug\\2007_000032.png"
# mmask = cv2.imread(path, 0)
# mmask[mmask > 0] = 255
# mmask[mmask != 255] = 0
# contours, hierarchy = cv2.findContours(mmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# print("mask info:", mmask.shape, len(contours))
#
# points = (100, 205)
# mask = cv2.circle(mmask, points, 4, (0, 255, 0), 4)
# cv2.imshow('img', mask)
# cv2.waitKey()
# all_points = 100
# in_points = 0
# dst = []
# for i in range(len(contours)):
#     dst.append(cv2.pointPolygonTest(np.array(contours[i]), points, 1))
#     if cv2.pointPolygonTest(np.array(contours[i]), points, 1) >= 0:
#         in_points = in_points + 1
#
# print(in_points)
#
#
# print(dst)

MAX_VALUE = 100

def update(img):
    """
    :param input_img_path
    :param output_img_path
    :param lightness
    :param saturation
    """

    # image = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    lightness = -50
    saturation = -50

    img = img.astype(np.float32) / 255.0
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1

    hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)

    return lsImg

def inorout_point(name, kpt, img):
    img = update(img)
    # path = "L:\\dataset\\VOC2012_\\VOC2012_att\\SegmentationClassAug\\"
    # path = path + name.replace(".jpg", ".png")
    # print(path)
    # mmask = cv2.imread(path, 0)
    # mmask = cv2.resize(mmask, (640, 480))
    # mmask[mmask > 0] = 255
    # mmask[mmask != 255] = 0
    # contours, hierarchy = cv2.findContours(mmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print("mask info:", mmask.shape, len(contours))

    points = kpt
    # mask = cv2.circle(mmask, points, 4, (0, 255, 0), 4)
    # cv2.imshow('img', mask)
    # cv2.waitKey()
    all_points = kpt.shape[0]

    # if all_points == 0:
    #     scale = 0
    #     print("单幅图像in_point个数：", 0)
    # else:
    #     in_points = 0
    #     for i in range(len(contours)):
    #         for j in range(all_points):
    #             dst = cv2.pointPolygonTest(np.array(contours[i]), points[j], 1)
    #             if dst >= 0:
    #                 in_points = in_points + 1



    # if all_points == 0:
    #     scale = 0
    #     print("单幅图像in_point个数：", 0)
    # else:
    #     in_points = 0
    #     for j in range(all_points):
    #         dst_0 = []
    #         for i in range(len(contours)):
    #             dst = cv2.pointPolygonTest(np.array(contours[i]), points[j], 1)
    #             dst_0.append(dst)
    #         if (np.array(dst_0) >= 0).any():
    #             points_img = cv2.circle(img, (int(points[j][0]), int(points[j][1])), 2, (0, 255, 0), -1)
    #             in_points = in_points + 1
    #         else:
    #             points_img = cv2.circle(img, (int(points[j][0]), int(points[j][1])), 2, (0, 0, 255), -1)
    #
    #     if in_points == 0 :
    #         points_img = img
    #
    #     print("in_point of single image：", in_points)
    #     print("all_point of single image：", all_points)
    # # print(all_points)
    # if all_points == 0 or in_points == 0:
    #     scale = 0
    # else:
    #     scale = in_points/all_points

    for j in range(all_points):
        points_img = cv2.circle(img, (int(points[j][0]), int(points[j][1])), 2, (255, 255, 1), -1)

    return points_img