import glob

import cv2
import numpy as np
import os

from matplotlib import cm
import re #regexpr
from PyQt4 import QtGui,QtCore

COLORMAP = 'flag'
COLORMAP_MAX = 200
MASK_POSTFIX = '_mask'
MARKER_POSTFIX = '_marker'
PREFIX_POSTFIX_SPLIT_SIGN = '_'


#TODO, hand over which segmentation tool should be used with which parameters when computeWatershed() is called
class segmentIt():
    def __init__(self, watershedParam):
        self._watershedParam = watershedParam

    def watershedSegmentation(self,img, markers):
        m = (markers+1).copy().astype(dtype=np.int32) # watershed takes class 0 as background
        cv2.watershed(img,m) #0 is considered uncertain
        m = m - 1  #bring labels back to normal
        m[m<0]=255
        m=m.astype(np.uint8)
        print 'Border pixels: %d' %  np.argwhere(m == 255).shape[0]
        return m


        #img[markers == -1] = [255, 0, 0] # to get the boundaries

def classToColorTuple(i):
    colormap = cm.ScalarMappable(cmap=COLORMAP)
    colormap.set_clim(0, COLORMAP_MAX)
    colorsRGB = colormap.to_rgba(i)
    return (int(colorsRGB[0] * 255), int(colorsRGB[1] * 255), int(colorsRGB[2] * 255))


def convertARGBarrayToQImage(npArray):
    height, width, channel = npArray.shape
    assert channel == 4, "this function only works for alpha channel including RGB arrays"
    assert npArray.dtype == np.uint8, "this function only works for uint8 arrays"
    perLine = channel * width
    npArray = cv2.cvtColor(npArray,cv2.COLOR_RGBA2BGRA)
    return QtGui.QImage(npArray.data, width, height, perLine, QtGui.QImage.Format_ARGB32)

def convertQImgToArray(qImg):
    width = qImg.width()
    height = qImg.height()
    buf = qImg.bits().asstring(qImg.numBytes())
    arr = np.frombuffer(buf, np.uint8).reshape(height,width,4)
    return cv2.cvtColor(arr,cv2.COLOR_BGRA2RGBA)


class ImageSet():
    def __init__(self,name, imgFile,  markerFile):
        self.name = name
        self.imgFile = imgFile
        self.markerFile = markerFile

def find_filter(imageSetList, name):
    return filter(lambda s: s.name == name, imageSetList)

#@brief communication towards the backbone is done via the public fcts
# comm leaving the backbone is done via SIGNALS
class ImageBackbone(QtCore.QObject):

    imgLoaded =  QtCore.pyqtSignal(QtGui.QImage, int, QtCore.QString, int, int) #img,id,imgName,minId,maxId
    watershedUpdate =  QtCore.pyqtSignal(QtGui.QImage)
    markersLoaded =  QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self,path, maxClasses, imgWidth, imgHeight, backgroundAlpha,
                 markerAlpha, watershedAlpha, markerNeutralColor, markerNeutralClass):
        """

        :type markerNeutralClass: int
        """
        super(ImageBackbone,self).__init__()
        #TODO load all images from path, display first
        self.path = path
        self._maxClasses = maxClasses
        self.maxId = -1
        self.minId = 1
        self._scaleFactorX= 1.
        self._scaleFactorY= 1.
        self._markerArray = None
        self._maskArray = None #class indexed image
        self._imageWidth = imgWidth
        self._imageHeight = imgHeight
        self._backgroundAlpha = backgroundAlpha
        self._markerAlpha = markerAlpha
        self._watershedAlpha = watershedAlpha
        self._markerNeutralColor = markerNeutralColor
        self._markerNeutralClass = markerNeutralClass

        #do filereading stuff

        #reconstruct state of work from a given directory
        self.imageSetList = list()
        lookUpLaterMarkerList = list()
        for filename in sorted(glob.glob('%s/*.png' % path )):  # assuming png
            markerString = re.findall('.*%s.*' % MARKER_POSTFIX, filename)
            maskString = re.findall('.*%s.*' % MASK_POSTFIX, filename)
            rawFilename = os.path.basename(filename)
            name = rawFilename.split('.')[0]
            if not markerString and not maskString:
                #we found the raw filename, push back new element
                self.imageSetList.append(ImageSet(name,filename,None))
            #TODO throw exception if _ in filename
            if markerString:
                name = rawFilename.split(MARKER_POSTFIX)[0]
                lookUpLaterMarkerList.append((name,filename))
        for name, filename in lookUpLaterMarkerList:
            matchingNameElements = find_filter(self.imageSetList,name)
            if matchingNameElements:
                matchingNameElements[0].markerFile = filename

        if self.imageSetList:
            self.maxId = len(self.imageSetList)
            self.minId = 0
            self._imgID = 0
        else:
            #TODO emit FATAL signal, no file loaded
            print 'not implemented: no file loaded'

    def convertIndexArrayToARGBArray(self, arr, alphaVal):
        assert len(arr.shape) == 2, "only supports single channel uint8 class index imgs"
        rows, cols = arr.shape
        retArr = np.zeros((rows, cols, 4), dtype=np.uint8)
        for i in range(0, self._maxClasses):
            classClrTpl = np.array(classToColorTuple(i))
            classIds = np.argwhere(arr == i)
            retArr[classIds[:, 0], classIds[:, 1], :] = np.hstack((classClrTpl, alphaVal))

        # neutral marker value
        classClrTpl = np.array(self._markerNeutralColor)
        classIds = np.argwhere(arr == self._markerNeutralClass)
        retArr[classIds[:, 0], classIds[:, 1], :] = np.hstack((classClrTpl, 0))  # make thisfully opaque

        return retArr

    def convertARGBArrayToIndexArray(self, arr):
        # multiply to get a single index array
        clrMultiplier = np.array([1, 2, 3])
        singleArr = np.dot(arr[:, :, :3], clrMultiplier)
        retArr = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
        # convert indexes
        for i in range(0, self._maxClasses):
            classClrTpl = np.array(classToColorTuple(i))
            clrWeigthedProduct = np.dot(classClrTpl, clrMultiplier)
            classIds = np.argwhere(singleArr == clrWeigthedProduct)
            retArr[classIds[:, 0], classIds[:, 1]] = i

        # neutral marker value
        classClrTpl = np.array(self._markerNeutralColor)
        clrWeigthedProduct = np.dot(classClrTpl, clrMultiplier)
        classIds = np.argwhere(singleArr == clrWeigthedProduct)
        retArr[classIds[:, 0], classIds[:, 1]] = self._markerNeutralClass
        return retArr


    def getNextImg(self):
        self.loadImg(self._imgID + 1)

    def getPreviousImg(self):
        self.loadImg(self._imgID - 1)

    def npArrayToQImage(self,img):
        height, width, channel = img.shape
        assert channel == 4
        bytesPerLine = channel * width
        return QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_ARGB32)


    def loadImg(self, i):
        if i < self.maxId and i >= self.minId:
            self._imgID = i
            #read the image
            filename = self.imageSetList[self._imgID].imgFile
            print 'INFO: Now reading %s' % filename
            if float(cv2.__version__.split('.')[0]) >= 3:
                self.img = cv2.imread(filename, cv2.IMREAD_COLOR)
            else:
                self.img = cv2.imread(filename,cv2.CV_LOAD_IMAGE_COLOR)
            if self.img is None:
                self.imageSetList.remove(self.imageSetList[self._imgID])
                self.maxId -=1
                return False
            self._scaleFactorX = self._imageHeight / float(self.img.shape[0])
            self._scaleFactorY = self._imageWidth / float(self.img.shape[1])
            #note, cv2 convention is x is horizontal axis, y is vertical, but for matrices its X x Y x M
            self.img = cv2.resize(self.img, None, fy=self._scaleFactorX , fx=self._scaleFactorY,
                                  interpolation=cv2.INTER_AREA)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2BGRA)
            self.img[:,:,3] = self._backgroundAlpha  # TODO, let the scribble widget set the alphas
            name = self.imageSetList[self._imgID].name
            self.imgLoaded.emit(self.npArrayToQImage(self.img), self._imgID, name, self.minId, self.maxId)
            if self.imageSetList[self._imgID].markerFile:
                if float(cv2.__version__.split('.')[0]) >= 3:
                    self._markerArray = cv2.imread(self.imageSetList[self._imgID].markerFile,
                                                   cv2.IMREAD_GRAYSCALE)
                else:
                    self._markerArray = cv2.imread(self.imageSetList[self._imgID].markerFile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                rgbArray = self.convertIndexArrayToARGBArray(self._markerArray, self._markerAlpha )
                self.markersLoaded.emit(convertARGBarrayToQImage(rgbArray))
                self.computeWatershedUpdate()
            else:
                self._markerArray = np.ones((self._imageHeight,self._imageWidth))*self._markerNeutralClass
                rgbArray = self.convertIndexArrayToARGBArray(self._markerArray, self._markerAlpha )
                self.markersLoaded.emit(convertARGBarrayToQImage(rgbArray))

    def save(self):
        markerFilename = os.path.join(os.path.dirname(self.imageSetList[self._imgID].imgFile), '%s%s.png' %
                                      (self.imageSetList[self._imgID].name,
                                         MARKER_POSTFIX) )
        maskFilename = os.path.join(os.path.dirname(self.imageSetList[self._imgID].imgFile), '%s%s.png' %
                                    (self.imageSetList[self._imgID].name,
                                         MASK_POSTFIX) )
        if self._markerArray is not None:
            cv2.imwrite(markerFilename,self._markerArray)
            self.imageSetList[self._imgID].markerFile = markerFilename
            print 'saved marker img to %s' % markerFilename
            #TODO emit signal safed successfully
        if self._maskArray is not None:
            maskResized = cv2.resize(self._maskArray, None, fy=1./self._scaleFactorX , fx=1./self._scaleFactorY,
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(maskFilename, maskResized)
            print 'saved segmentation mask img to %s' % maskFilename
            #TODO emit signal safed successfully

    # TODO, at some point do not allow to do too many recalcs
    def computeWatershedUpdate(self):
        sI = segmentIt(1)
        tmpImg = cv2.cvtColor(self.img,cv2.COLOR_RGBA2BGR)
        #TODO, hand over which segmentation tool should be used with which parameters when computeWatershed() is called
        self._maskArray = sI.watershedSegmentation(tmpImg,self._markerArray)
        rgbArray = self.convertIndexArrayToARGBArray(self._maskArray, self._watershedAlpha)  # TODO, let the scribble widget set the alphas
        self.watershedUpdate.emit(convertARGBarrayToQImage(rgbArray))

    def markerUpdate(self, markerQImg):
        rgbArray = convertQImgToArray(markerQImg)
        self._markerArray = self.convertARGBArrayToIndexArray(rgbArray)




#@brief borrowed a lot from http://stackoverflow.com/questions/6542141/painting-using-pyqt
class ScribbleImage(QtGui.QWidget):
    markerUpdate = QtCore.pyqtSignal()

    def __init__(self, markerAlpha, markerNeutralColor, imgWidth, imgHeight, parent=None):
        super(ScribbleImage, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.scribbling = False
        self.lining = False
        self._penWidth = 3
        self._imgWidth = imgWidth
        self._imgHeight = imgHeight
        self._markerNeutralColor = markerNeutralColor
        self._markerAlpha = markerAlpha #TODO make interactively settable
        self.lastPoint = QtCore.QPoint()
        self.backgroundImage = None
        self.watershedImg = None
        self.markerImgRGB = None
        self._showWatershed = False
    #expects a color tuple
    def setPenColor(self, newColor, markerAlpha):
        self._penColor = QtGui.QColor(newColor[0], newColor[1], newColor[2], markerAlpha)

    def setPenWidth(self, newWidth):
        self._penWidth = newWidth


    def setEraseMode(self):
        self.setPenColor(self._markerNeutralColor, 1 ) #small trick. this is not seen. but the qpainter paints (0,0,0)
        # color automagically, when 0 opacity is set
        self.setPenWidth(15)

    def setActiveMarkerClass(self, classId):
        self.setPenColor(classToColorTuple(classId), self._markerAlpha)
        self.setPenWidth(3)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True
#        if event.button() == QtCore.Qt.RightButton:
#            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False
        if event.button() == QtCore.Qt.RightButton:
            self.drawLineTo(event.pos(), directLine=True)
        self.lastPoint = event.pos()
        self.markerUpdate.emit()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        imageWithOverlay = QtGui.QImage(self.markerImgRGB.size(), QtGui.QImage.Format_ARGB32_Premultiplied)
        #
        if self.backgroundImage:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            painter.drawImage(imageWithOverlay.rect(), self.backgroundImage)
        if self.watershedImg and self._showWatershed:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationOver)
            painter.drawImage(imageWithOverlay.rect(), self.watershedImg)

        if self.markerImgRGB:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            painter.drawImage(imageWithOverlay.rect(), self.markerImgRGB)


    def drawLineTo(self, endPoint,directLine=False):
        painter = QtGui.QPainter(self.markerImgRGB)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source) #overwrites hard instead of merging by alphas or whatever
        painter.setPen(QtGui.QPen(self._penColor, self._penWidth,
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)

        #TODO, use drawlines to buffer x lines and enable undo
        if self.lastPoint == endPoint:
            painter.drawPoint(self.lastPoint)
        self.update()
        self.lastPoint = QtCore.QPoint(endPoint)


    def penColor(self):
        return self._penColor


    def penWidth(self):
        return self._penWidth

    def clearImage(self):
        if self.markerImgRGB:
            self.markerImgRGB.fill(QtGui.qRgba(self._markerNeutralColor[0], self._markerNeutralColor[1],
                                                        self._markerNeutralColor[2], 0))  # make fully opaque
            self.update()

    def slot_setMarkerImage(self,qImgRGB):
        self.markerImgRGB = qImgRGB.copy()
        self.update()

    def slot_setBackgroundImage(self,qImgRGB,i , s, i2, i3):
        self.backgroundImage = qImgRGB.copy()
        self.update()


    def slot_setWatershedImg(self, qImg):
        self.watershedImg = qImg.copy()
        self.update()

    def showWatershed(self, setActive):
        self._showWatershed = setActive
        self.update()