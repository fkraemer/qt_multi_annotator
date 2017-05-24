#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PyQt4 import QtCore, QtGui
import numpy as np

from annotation_helper import *
import unittest
import os
import pickle
import argparse
#for file listing from path

CLASSBUTTON_MIN_HEIGHT = 30
CLASSBUTTON_MIN_WIDTH = 40
MAX_CLASSES = 20
IMG_SIZE_X = 800
IMG_SIZE_Y = 1024
MARKER_NEUTRAL_COLOR = (255, 255, 255)
MARKER_NEUTRAL_CLASS = 255
DEFAULT_MARKER_ALPHA = 255
DEFAULT_WATERSHED_ALPHA = 220
DEFAULT_BACKGROUND_ALPHA = 155
DEFAULT_CLASS_NUM = 5
CLASSNAMES_FILENAME = 'classnames.txt'


parser = argparse.ArgumentParser(description='Label a folders files')
parser.add_argument('path', type=str,
                    help='where to look for images, non-recursive')
parser.add_argument('--classes', type=str,
                    help= ('class names seperated by commas: foo1,bar2,yala (max: %d)' % MAX_CLASSES) )


#@brief overwrite to pass the class id when clicked with signal "clickedClass"
class flosQPushButton(QtGui.QPushButton):
    clickedClass =  QtCore.pyqtSignal(int)

    def __init__(self, _str, id):
        super(flosQPushButton,self).__init__(_str)
        self.classId = id
        QtCore.QObject.connect(self, QtCore.SIGNAL('clicked()'), self.clickedSignal)

    def clickedSignal(self):
        self.clickedClass.emit(self.classId)


class AnnotationWindow(QtGui.QWidget):
    def __init__(self, imgBackbone, nameList):
        """
        for every entry in nameList a labelling button is created
        :type imgBackbone: ImageBackbone
        :type nameList: list        
        """
        QtGui.QWidget.__init__(self)
        self.imgBackbone = imgBackbone
        self.classMax = classMax
        self.watershedActivated = False
        self.activeClassButtonId = 0
        self.nameList = nameList
        self.goToChangeSlotActive = True

        self.scribbleArea = ScribbleImage(DEFAULT_MARKER_ALPHA, MARKER_NEUTRAL_COLOR, IMG_SIZE_Y, IMG_SIZE_X, parent=self)
        self.scribbleArea.setActiveMarkerClass(self.activeClassButtonId)

        self.setupUI()
        self.connectThings()
        imgBackbone.loadImg(0)

    def connectThings(self):
        # GUI stuff
        #Left
        QtCore.QObject.connect(self.nextButton, QtCore.SIGNAL('clicked()'), self.slot_openNextImg) #this forwards basically to goTo valuechanged signal
        QtCore.QObject.connect(self.previousButton, QtCore.SIGNAL('clicked()'), self.slot_openPreviousImg) #this forwards basically to goTo valuechanged signal
        QtCore.QObject.connect(self.goToWidget, QtCore.SIGNAL('valueChanged(int)'), self.slot_openImage)
        #Middle
        QtCore.QObject.connect(self.scribbleArea, QtCore.SIGNAL('markerUpdate()'), self.slot_computeWatershed)
        for d in self.classButtonList:
            QtCore.QObject.connect(d, QtCore.SIGNAL('clickedClass(int)'), self.slot_classButtonPushed)
        QtCore.QObject.connect(self.eraseMarkerButton, QtCore.SIGNAL('clicked()'), self.slot_eraseButtonPushed)
        #Right
        QtCore.QObject.connect(self.exitButton, QtCore.SIGNAL('clicked()'), self.slot_closeHandle)
        QtCore.QObject.connect(self.saveButton, QtCore.SIGNAL('clicked()'), self.slot_save)
        QtCore.QObject.connect(self.saveAndNextButton, QtCore.SIGNAL('clicked()'), self.slot_saveAndNext)
        QtCore.QObject.connect(self.clearMarkerButton, QtCore.SIGNAL('clicked()'), self.slot_clearMarkerImage)
        QtCore.QObject.connect(self.segmentationActiveWidget, QtCore.SIGNAL('stateChanged(int)'), self.slot_watershedActiveChange)

        #Background tasks
        QtCore.QObject.connect(self.imgBackbone, QtCore.SIGNAL('imgLoaded(QImage, int , QString, int, int)'), self.slot_updateImg)
        QtCore.QObject.connect(self.imgBackbone, QtCore.SIGNAL('imgLoaded(QImage, int , QString, int, int)'),
                               self.scribbleArea.slot_setBackgroundImage)
        QtCore.QObject.connect(self.imgBackbone, QtCore.SIGNAL('watershedUpdate(QImage)'), self.scribbleArea.slot_setWatershedImg)
        QtCore.QObject.connect(self.imgBackbone, QtCore.SIGNAL('markersLoaded(QImage)'), self.scribbleArea.slot_setMarkerImage)

    def setupUI(self):

        #Left Column
        self.currentImgLabel = QtGui.QLabel("ImageName \n 0 / N")
        self.goToWidget = QtGui.QSpinBox() #configured in button update
        self.previousButton = QtGui.QPushButton("<<")
        self.nextButton = QtGui.QPushButton(">>")

        vboxLeft = QtGui.QVBoxLayout()
        vboxLeft.setAlignment(QtCore.Qt.AlignTop)
        vboxLeft.addWidget(self.currentImgLabel)
        vboxLeft.addWidget(self.goToWidget)
        vboxLeft.addWidget(self.previousButton)
        vboxLeft.addWidget(self.nextButton)

        #Middle Column
        vboxMiddle = QtGui.QVBoxLayout()
        hboxMiddle = QtGui.QHBoxLayout()
        hboxMiddle.setAlignment(QtCore.Qt.AlignLeft)
        currentClassLabel = QtGui.QLabel("Class:")
        hboxMiddle.addWidget(currentClassLabel)

        #upper row:                                BUTTONS
        self.classButtonList = list()
        for i in range(0,self.classMax):
            qP = flosQPushButton("%s" % self.nameList[i],i )
            qP.setMinimumWidth(CLASSBUTTON_MIN_WIDTH)
            qP.setMinimumHeight(CLASSBUTTON_MIN_HEIGHT)
            self.classButtonList.append(qP)
            hboxMiddle.addWidget(qP)
        self.eraseMarkerButton = QtGui.QPushButton("Erase")
        self.eraseMarkerButton.setMinimumWidth(CLASSBUTTON_MIN_WIDTH*2)
        self.eraseMarkerButton.setMinimumHeight(CLASSBUTTON_MIN_HEIGHT)
        hboxMiddle.addWidget(self.eraseMarkerButton)
        self.guiActivateClassButton()

        vboxMiddle.addLayout(hboxMiddle)

        #lower part: image
        self.scribbleArea.setMinimumHeight(IMG_SIZE_X)
        self.scribbleArea.setMinimumWidth(IMG_SIZE_Y)
        vboxMiddle.addWidget(self.scribbleArea)

        ##Right Column Sub groups

        ###watershed params
        vboxRightWatershedSubLayout = QtGui.QVBoxLayout()
        self.segmentationActiveWidget = QtGui.QCheckBox("Paint Segm.")
        # self.watershedOptions = ['Normal','Foo','Bar']
        # self.watershedEdit = QtGui.QComboBox()
        # self.watershedEdit.addItems(self.watershedOptions)
        # vboxRightWatershedSubLayout.addWidget(self.watershedEdit)
        vboxRightWatershedSubLayout.addWidget(self.segmentationActiveWidget)
        watershedOptionsGroup = QtGui.QGroupBox("Segmentation")
        # watershedGroup.setStyleSheet('border: 1px solid black; border-radius: 5px; margin-top: 1ex')
        watershedOptionsGroup.setLayout(vboxRightWatershedSubLayout)

        ###save params
        self.clearMarkerButton = QtGui.QPushButton("Clear Markers")
        self.saveButton = QtGui.QPushButton("Save")
        self.saveAndNextButton = QtGui.QPushButton('Save + \n Next')
        self.saveAndNextButton.setMinimumHeight(100)
        self.exitButton = QtGui.QPushButton('Exit')
        self.exitButton.setMinimumHeight(100)
        vboxRightSaveSubLayout = QtGui.QVBoxLayout()
        vboxRightSaveSubLayout.addWidget(self.clearMarkerButton)
        vboxRightSaveSubLayout.addWidget(self.saveButton)
        vboxRightSaveSubLayout.addWidget(self.saveAndNextButton)
        vboxRightSaveSubLayout.addWidget(self.exitButton)
        imageOptionsGroup = QtGui.QGroupBox("Image")
        # watershedGroup.setStyleSheet('border: 1px solid black; border-radius: 5px; margin-top: 1ex')
        imageOptionsGroup.setLayout(vboxRightSaveSubLayout)

        # Right Column Top Layout
        vboxRight = QtGui.QVBoxLayout()
        vboxRight.addStretch(1)
        vboxRight.addWidget(watershedOptionsGroup)
        vboxRight.addWidget(imageOptionsGroup)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vboxLeft)
        hbox.addLayout(vboxMiddle)
        hbox.addLayout(vboxRight)
        self.setLayout(hbox)

        self.setWindowTitle('Multi Class Annotation Tool')
        self.show()


    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Q:
            self.slot_closeHandle()
        if e.key() == QtCore.Qt.Key_S:
            self.slot_save()
        if e.key() == QtCore.Qt.Key_T:
            self.segmentationActiveWidget.toggle()
        print e.key()

    def guiActivateClassButton(self):
        deactivatedButtonList = range(0,len(self.classButtonList))
        try:
            deactivatedButtonList.remove(self.activeClassButtonId)
        except ValueError:
            pass # it was just the erase class
        for i in deactivatedButtonList:
            self.classButtonList[i].setStyleSheet(
                "background-color: rgb(%d, %d, %d);  border: none" % classToColorTuple(i))
        #extra handling of erase mode
        if self.activeClassButtonId == MARKER_NEUTRAL_CLASS: # we are in erase mode
            self.eraseMarkerButton.setStyleSheet("background-color: white;  border: 3px solid black")
        else:
            self.eraseMarkerButton.setStyleSheet("background-color: white;  border: 3px solid red")
            self.classButtonList[self.activeClassButtonId].setStyleSheet(
                "background-color: rgb(%d, %d, %d);  border: 3px solid black" % classToColorTuple(self.activeClassButtonId))


    def guiButtonUpdate(self,currentId,minId,maxId):
        self.goToWidget.setMaximum(maxId-1)
        self.goToWidget.setMinimum(minId)
        self.goToWidget.setKeyboardTracking(False)
        #the goto widgets value change should not provoke a load request
        self.goToChangeSlotActive = False
        self.goToWidget.setValue(currentId)
        self.goToChangeSlotActive = True
        if currentId == (maxId-1):
            self.nextButton.setEnabled(False)
            self.saveAndNextButton.setEnabled(False)
        else:
            self.nextButton.setEnabled(True)
            self.saveAndNextButton.setEnabled(True)
        if currentId == minId:
            self.previousButton.setEnabled(False)
        else:
            self.previousButton.setEnabled(True)

    def saveThisImageAndMask(self):
        self.imgBackbone.markerUpdate(self.scribbleArea.markerImgRGB)
        self.imgBackbone.computeWatershedUpdate()
        self.imgBackbone.save()

    def slot_openImage(self,id):
        if self.goToChangeSlotActive:
            self.imgBackbone.loadImg(id)

    def slot_openNextImg(self):
        self.imgBackbone.getNextImg()

    def slot_openPreviousImg(self):
        self.imgBackbone.getPreviousImg()

    def slot_closeHandle(self):
        self.close()

    def slot_save(self):
        self.saveThisImageAndMask()

    def slot_saveAndNext(self):
        self.saveThisImageAndMask()
        self.slot_openNextImg()

    def slot_updateImg(self,qImg,id,imgName,minId,maxId):
        #set new id
        self.guiButtonUpdate(id,minId,maxId)
        self.currentImgLabel.setText('%s \n %3d / %3d' %(imgName,id,maxId-1))

    def slot_clearMarkerImage(self):
        self.scribbleArea.clearImage()
        if self.watershedActivated:
            self.segmentationActiveWidget.setChecked(False)
            #TODO test, whether this emits the signal and provokes a watershed update

    def slot_idNotValid(self,id,minId, maxId):
        #TODO show dialog
        print '%d is not a valid image id, must be between %d and %d' % (id,minId, maxId,)

    def slot_computeWatershed(self):
        #TODO do some timer stuff here, to not do it too often
        if self.watershedActivated:
            self.imgBackbone.markerUpdate(self.scribbleArea.markerImgRGB)
            self.imgBackbone.computeWatershedUpdate()

    def slot_watershedActiveChange(self,i):
        if i > 0:
            self.watershedActivated = True
            self.scribbleArea.showWatershed(True)
            self.slot_computeWatershed()
        else:
            self.watershedActivated = False
            self.scribbleArea.showWatershed(False)

    def slot_classButtonPushed(self,classId):
        self.activeClassButtonId = classId
        self.guiActivateClassButton()
        self.scribbleArea.setActiveMarkerClass(classId)

    def slot_eraseButtonPushed(self):
        self.activeClassButtonId = MARKER_NEUTRAL_CLASS #this will 'unset' all other buttons
        self.guiActivateClassButton()
        self.scribbleArea.setActiveMarkerClass(2)
        self.scribbleArea.setEraseMode()

def loadClassNamesOrSetNew(path, rawNames):
    path = os.path.join(path,CLASSNAMES_FILENAME)
    if os.path.exists(path) and not rawNames:
        with open(path, 'rb') as f:
            names = pickle.load(f)
    elif rawNames:
        names = rawNames.split(',')
        with open(path, 'wb') as f:
            pickle.dump(names, f)
    else:
        names = [ str(i) for i in range(0,DEFAULT_CLASS_NUM)]
    return names

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    args = parser.parse_args()
    nameList = loadClassNamesOrSetNew(args.path,args.classes)
    # nameList = ['Boden', 'Wasser', 'Bodenholz', 'Tanne', 'Fichte', 'Kiefer', 'Blattlos', 'DontCare']
    classMax = np.min((len(nameList),MAX_CLASSES)) #limit this
    nameList = nameList[0:classMax]
    imgBackbone = ImageBackbone(args.path,MAX_CLASSES,IMG_SIZE_Y,IMG_SIZE_X,#let the backbone always work up to MAX_CLASSES so that wrong user input can not destroy once set labels
                                DEFAULT_BACKGROUND_ALPHA, DEFAULT_MARKER_ALPHA, DEFAULT_WATERSHED_ALPHA,
                                MARKER_NEUTRAL_COLOR,MARKER_NEUTRAL_CLASS)
    myapp = AnnotationWindow(imgBackbone,nameList)
    myapp.show()
    sys.exit(app.exec_())


class test(unittest.TestCase):
    def testImageSetFilling(self):
        s = 'foo'
        tmpDir = 'test_tmp'
        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)
        fileList = ['img001.png', 'img001_marker.png','img002.png','img003.png', 'img003_marker.png',
                    'img004_marker.png']
        for fl in fileList:
            filename = '%s/%s' % (tmpDir,fl)
            with open(filename, "w") as f:
                f.write("FOOBAR")
        imgBackbone = ImageBackbone(tmpDir,MAX_CLASSES,IMG_SIZE_Y,IMG_SIZE_X,#let the backbone always work up to MAX_CLASSES so that wrong user input can not destroy once set labels
                                DEFAULT_BACKGROUND_ALPHA, DEFAULT_MARKER_ALPHA, DEFAULT_WATERSHED_ALPHA,
                                MARKER_NEUTRAL_COLOR,MARKER_NEUTRAL_CLASS)
        #test assertions
        self.assertEqual(len(imgBackbone.imageSetList),3)
        imgSetList = imgBackbone.imageSetList
        self.assertIsNotNone(find_filter(imgSetList,'img001'))
        self.assertIsNotNone(find_filter(imgSetList,'img003')[0].markerFile)
        self.assertEqual(len(find_filter(imgSetList,'img004')),0)
        #clean up
        for fl in fileList:
            os.remove('%s/%s' % (tmpDir,fl))
        os.removedirs(tmpDir)

    def testQImageConversion(self):
        imgBackbone = ImageBackbone('foo',MAX_CLASSES,IMG_SIZE_Y,IMG_SIZE_X,#let the backbone always work up to MAX_CLASSES so that wrong user input can not destroy once set labels
                                DEFAULT_BACKGROUND_ALPHA, DEFAULT_MARKER_ALPHA, DEFAULT_WATERSHED_ALPHA,
                                MARKER_NEUTRAL_COLOR,MARKER_NEUTRAL_CLASS)
        testImg = np.zeros((IMG_SIZE_X, IMG_SIZE_Y, 4), np.uint8)
        testImg[20:100,20:100,:] = 10
        testImg[20:-100,20:-100,:] = 255
        qImg = convertARGBarrayToQImage(testImg)
        testImgConverted = convertQImgToArray(qImg)
        self.assertTrue( np.all(testImg.shape==testImgConverted.shape) )
        self.assertTrue( np.all(testImg==testImgConverted) )

        # a = convertQImgToArray(self.markerImgRGB)
        # b = convertARGBArrayToIndexArray(a)
        # c = convertIndexArrayToARGBArray(b,150)
        # d = convertARGBarrayToQImage(c)


        indexImg = np.zeros((IMG_SIZE_X, IMG_SIZE_Y), np.uint8)
        indexImg[20:-100,20:100] = 10
        indexImg[20:-100, 20:100] = 1
        rgbArr = imgBackbone.convertIndexArrayToARGBArray(indexImg, DEFAULT_MARKER_ALPHA)
        indexCmp = imgBackbone.convertARGBArrayToIndexArray(rgbArr)
        self.assertTrue(np.all(indexImg.shape == indexCmp.shape))
        self.assertEqual(np.argwhere(indexImg != indexCmp).shape[0],0)