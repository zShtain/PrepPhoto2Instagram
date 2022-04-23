import cv2
from matplotlib import pyplot as plt
from numpy import zeros, arange, meshgrid, array, logical_or, nonzero
from pathlib import Path
import argparse
import sys


def createMaps(srcWidth, srcHeight, dstWidth=2048, dstHeight=2048):
    x = arange(dstWidth)
    y = arange(dstHeight)
    x, y = meshgrid(x, y)

    t0 = array([-dstWidth / 2.0, -dstHeight / 2.0])
    t1 = array([srcWidth / 2.0, srcHeight / 2.0])
    s = srcWidth / dstWidth if srcWidth > srcHeight else srcHeight / dstHeight
    s2 = 0.6 * s
    s3 = srcHeight / (0.9 * dstHeight)
    t2 = array([-dstWidth, -dstHeight / 2.0])
    t3 = array([0, -dstHeight / 2.0])

    x1 = (t1[0] + s * (x.reshape(-1) + t0[0])).reshape((dstHeight, dstWidth))
    y1 = (t1[1] + s * (y.reshape(-1) + t0[1])).reshape((dstHeight, dstWidth))
    x2 = (t1[0] + s2 * (x.reshape(-1) + t0[0])).reshape((dstHeight, dstWidth))
    y2 = (t1[1] + s2 * (y.reshape(-1) + t0[1])).reshape((dstHeight, dstWidth))
    if srcWidth > srcHeight:
        x3 = (t1[0] + s3 * (x.reshape(-1) + t2[0])).reshape((dstHeight, dstWidth))
        y3 = (t1[1] + s3 * (y.reshape(-1) + t2[1])).reshape((dstHeight, dstWidth))
        x4 = (t1[0] + s3 * (x.reshape(-1) + t3[0])).reshape((dstHeight, dstWidth))
        y4 = (t1[1] + s3 * (y.reshape(-1) + t3[1])).reshape((dstHeight, dstWidth))
        return (array(x1, 'f'), array(y1, 'f'), array(x2, 'f'), array(y2, 'f'),
                array(x3, 'f'), array(y3, 'f'), array(x4, 'f'), array(y4, 'f'))

    else:
        return (array(x1, 'f'), array(y1, 'f'), array(x2, 'f'), array(y2, 'f'),
                None, None, None, None)


def instaImages(src, alpha=0.6, backgroundImg=None):

    dsts = []
    blur = cv2.GaussianBlur(src, (31, 31), 10)
    blur = cv2.addWeighted(blur, alpha, array(zeros(blur.shape), dtype='uint8'), 1 - alpha, 0)

    mapX1, mapY1, mapX2, mapY2, mapX3, mapY3, mapX4, mapY4 = createMaps(src.shape[1], src.shape[0])
    dst = cv2.remap(src, mapX1, mapY1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    blur = cv2.remap(blur, mapX2, mapY2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    if backgroundImg is not None:
        blur = cv2.addWeighted(blur, alpha, backgroundImg, 1 - alpha, 0)
    dst[dst == 0] = blur[dst == 0]
    dsts.append(dst)

    if not(mapX3 is None or mapY3 is None or mapX4 is None or mapY4 is None):
        dst1 = cv2.remap(src, mapX3, mapY3, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        dst2 = cv2.remap(src, mapX4, mapY4, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        if backgroundImg is None:
            dst1[logical_or(logical_or(mapX3 < 0, mapX3 > src.shape[1]), 
                            logical_or(mapY3 < 0, mapY3 > src.shape[0])), :] = 255
            dst2[logical_or(logical_or(mapX4 < 0, mapX4 > src.shape[1]), 
                            logical_or(mapY4 < 0, mapY4 > src.shape[0])), :] = 255
        else:
            tmp = nonzero(logical_or(logical_or(mapX3 < 0, mapX3 > src.shape[1]), 
                                     logical_or(mapY3 < 0, mapY3 > src.shape[0])))
            dst1[tmp[0], tmp[1], :] = bImg[tmp[0], tmp[1], :]
            tmp = nonzero(logical_or(logical_or(mapX4 < 0, mapX4 > src.shape[1]), 
                                     logical_or(mapY4 < 0, mapY4 > src.shape[0])))
            dst2[tmp[0], tmp[1], :] = bImg[tmp[0], tmp[1], :]
        dsts.append(dst1)
        dsts.append(dst2)

    plt.figure(figsize=(8, 6))
    plt.imshow(dst)
    plt.xticks([])
    plt.yticks([])

    plt.figure(figsize=(8, 6))
    plt.subplot(132)
    plt.imshow(dst1)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    plt.imshow(dst2)
    plt.xticks([])
    plt.yticks([])

    return dsts


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputImage', default=None, required=True)
    parser.add_argument('--outputPath', default=None, required=False)

    inputImage = ""
    outPath = ""
    if len(sys.argv) > 1:
        args = parser.parse_args()

        if args.inputImage is None:
            raise ValueError('Missing input image')
        inputPath = args.inputImage
        outPath = "./output/" if args.outputPath is None else str(args.outputPath)

    else:
        inputImage = "C:/Users/Zachi/OneDrive/Desktop/New Folder/189A9403.jpg"
        outPath = "C:/Users/Zachi/OneDrive/Desktop/New Folder/output/"
        backgroundPath = "./input/InstaTemplate.jpg"

    imgPath = Path(inputImage)
    img = cv2.imread(str(imgPath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if backgroundPath != '':
        bImg = cv2.imread(backgroundPath)
        bImg = cv2.cvtColor(bImg, cv2.COLOR_BGR2RGB)
    else:
        bImg = None

    imgType = imgPath.suffix
    imgName = imgPath.name.replace(imgType, '')
    
    outImgs = instaImages(img, 0.6, bImg)
    for i in range(len(outImgs)):
        plt.imsave(outPath + '\\' + imgName + '_' + str(i + 1) + imgType, outImgs[i])

    plt.show()