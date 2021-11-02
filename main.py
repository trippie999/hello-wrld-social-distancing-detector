# import the necessary packages
from hellowrld.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import time
import cv2 as cv
import os
import sys
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *

#base path to YOLO directory
MODEL_PATH = "yolo-coco"
#initialize minimum probability to filter weak detections along with
#the threshold when applying non maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

#boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = True

#define minimnum safe distance in px
MIN_DISTANCE = 50

#monitor to display the screen alarm
display_monitor = 1

#video feed directory
FeedDirectory = "demo.avi"

#seconds of interval per frame check
intervalsec = 5


cv2 = cv
safecount = 6
globalviolate = 0
rescale = True


class Initialization(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Initialization")
        self.setStyleSheet("background-color: #e6e6ea;")
        self.setWindowIcon(QIcon('logo.png'))
        validator = QRegExpValidator(QRegExp(r'[0-9]+'))
        #SAFECOUNT
        self.setSafeCount = QLabel("Safe Count (default: {} people):".format(safecount))
        self.safeCountInput = QLineEdit()
        self.safeCountInput.setValidator(validator)

        #MIN_DISTANCE
        self.minDistLabel = QLabel("Minimum Distance (default: {}px):".format(MIN_DISTANCE))
        self.pixelinput = QLineEdit()
        self.pixelinput.setValidator(validator)

        self.intervalLabel = QLabel("Frame Check Interval (default: {} second(s)):".format(intervalsec))
        self.intervalInput = QLineEdit()
        self.intervalInput.setValidator(validator)

        #CHOOSE_MONITOR
        self.monitorLabel = QLabel("Screen Alarm Monitor output (default: Display {}):".format(display_monitor+1))
        self.monitorinput = QLineEdit()
        self.monitorinput.setValidator(validator)
        #APPLY CHANGES
        self.button = QPushButton('Apply')
        self.button.clicked.connect(self.applychanges)
        #FOOTAGE SETTINGS
        self.button1 = QPushButton('Upload footage')
        self.button1.clicked.connect(self.get_video_file)
        self.button2 = QPushButton('Run Software')
        self.button2.clicked.connect(self.run_software)
        self.button3 = QPushButton('Run with External Camera')
        self.button3.clicked.connect(self.run_software_camera)
        self.note = QLabel("note: pressing 'run software' without uploading\n a footage will run a demo video.".format(MIN_DISTANCE))
        #CUDA SETTINGS
        self.cudacheck = QCheckBox('Use NVIDIA CUDA Processing (Recommended)')
        self.cudacheck.setChecked(True)
        #RESCALING
        # CUDA SETTINGS
        self.rescalecheck = QCheckBox('Downscale Video Feed (better performance for footages above 480p)')
        self.rescalecheck.setChecked(True)

        #EXTRA
        pixmap = QPixmap('initscreen.png')
        smaller_pixmap = pixmap.scaled(300,300)
        self.initlogo = QLabel(self)
        self.initlogo.setPixmap(smaller_pixmap)
        self.initlogo.setGeometry(0,0,300,300)
        self.initlogo.setScaledContents(True)
        self.initlogo.setAlignment(Qt.AlignCenter)
        self.advancedTxt = QLabel("\nAdvanced Settings")
        self.hellowrld = QLabel("Social Distancing Detector")
        self.hellowrld.setAlignment(Qt.AlignCenter)
        self.hellowrld.setStyleSheet("font-weight: bold; color: 262626")


        layout = QVBoxLayout()
        layout.addWidget(self.initlogo)
        layout.addWidget(self.hellowrld)
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.note)
        layout.addWidget(self.advancedTxt)
        layout.addWidget(self.setSafeCount)
        layout.addWidget(self.safeCountInput)
        layout.addWidget(self.minDistLabel)
        layout.addWidget(self.pixelinput)
        layout.addWidget(self.intervalLabel)
        layout.addWidget(self.intervalInput)
        layout.addWidget(self.monitorLabel)
        layout.addWidget(self.monitorinput)
        layout.addWidget(self.cudacheck)
        layout.addWidget(self.rescalecheck)
        layout.addWidget(self.button)
        self.setLayout(layout)


    def applychanges(self,):
        global MIN_DISTANCE
        global display_monitor
        global USE_GPU
        global safecount
        global intervalsec
        global rescale

        if self.safeCountInput.text() == "":
            print("Safecount set to default ({} people).".format(safecount))
            pass
        else:
            safecount_input = self.safeCountInput.text()
            safecount = int(safecount_input)
            print("Safecount set to {} people.".format(safecount))

        if self.pixelinput.text() == "":
            print("Distance set to default ({} px.).".format(MIN_DISTANCE))
            pass
        else:
            pixval = self.pixelinput.text()
            MIN_DISTANCE = int(pixval)
            print("Distance set to {} px.".format(MIN_DISTANCE))

        if self.monitorinput.text() == "":
            print("Monitor Alarm set to default (Display {}).".format(int(display_monitor)))
            pass

        else:
            monitorval = self.monitorinput.text()
            print("Monitor Alarm set to Display {}.".format(int(monitorval)))
            display_monitor = int(monitorval)-1
            print(display_monitor)
            print(type(display_monitor))

        if self.intervalInput.text() == "":
            print("Framecheck Interval to default ({} second(s)).".format(intervalsec))
            pass
        else:
            interval_input = self.intervalInput.text()
            intervalsec = int(interval_input)
            print("Framecheck Interval set to {} second(s).".format(intervalsec))

        if self.cudacheck.isChecked():
            USE_GPU = True
            print("CUDA set to On.")
        else:
            USE_GPU = False
            print("CUDA set to Off.")

        if self.rescalecheck.isChecked():
            rescale = True
            print("Rescaling set to On.")
        else:
            rescale = False
            print("Rescaling set to Off.")
        self.settingsapplied()



    def settingsapplied(self):
        msg = QMessageBox()

        msg.setWindowTitle('Notification')
        msg.setWindowIcon(QIcon('logo.png'))
        msg.setText("Settings Applied.")
        x = msg.exec_()

    def footageselected(self):
        msg = QMessageBox()

        msg.setWindowTitle('Notification')
        msg.setWindowIcon(QIcon('logo.png'))
        msg.setText("Footage Selected.")
        x = msg.exec_()

    def get_video_file(self):
        global FeedDirectory
        file_name, _ = QFileDialog.getOpenFileName(self,'Open Video File', r"<Default dir>", "Video files (*.mp4 *.avi)")
        FeedDirectory = file_name
        self.footageselected()



    def run_software(self):
        self.main = MainWindow()
        self.main.show()



    def run_software_camera(self):
        global FeedDirectory
        FeedDirectory = 0
        self.main = MainWindow()
        self.main.show()

class PopWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #e6e6ea;")
        self.showMaximized()
        self.setWindowTitle("Screen Alarm")
        self.setWindowIcon(QIcon('logo.png'))

        self.VBL = QVBoxLayout()
        self.image = QLabel(self)
        self.image.setPixmap(QPixmap("alert.jpg"))
        self.image.setGeometry(0, 0, 1920, 1080)
        self.image.setScaledContents(True)
        self.image.show()


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #e6e6ea;")
        self.setWindowTitle("Social Distancing Detector")
        self.setWindowIcon(QIcon('logo.png'))
        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.label3 = QLabel("Number of Violations: ")
        self.VBL.addWidget(self.label3)

        self.violationlabel = QLCDNumber()
        self.VBL.addWidget(self.violationlabel)

        self.label1 = QLabel("Choose an Alarm:")
        self.VBL.addWidget(self.label1)

        self.comboBox = QComboBox()
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("No Alarm")
        self.comboBox.addItem("Screen Alarm")
        self.comboBox.addItem("Sound Alarm")
        self.comboBox.addItem("Both Alarms")
        self.VBL.addWidget(self.comboBox)
        self.comboBox.currentIndexChanged.connect(self.AlarmSelection)



        self.label2 = QLabel("Manual Alarms:")
        self.VBL.addWidget(self.label2)

        self.mScreen = QPushButton()
        self.mScreen.setText("Screen Alarm")
        self.mScreen.clicked.connect(self.ExternalWindow)
        self.VBL.addWidget(self.mScreen)

        self.mSound = QPushButton()
        self.mSound.setText("Sound Alarm")
        self.mSound.clicked.connect(self.SoundAlarm)
        self.VBL.addWidget(self.mSound)

        self.mBoth = QPushButton()
        self.mBoth.setText("Both Alarms")
        self.mBoth.clicked.connect(self.BothAlarm)
        self.VBL.addWidget(self.mBoth)


        self.externalClose = QPushButton()
        self.externalClose.setText("Close Screen Alarm Window")
        self.externalClose.clicked.connect(self.CloseScreen)
        self.VBL.addWidget(self.externalClose)

        self.configCheck = QPushButton()
        self.configCheck.setText("Check Current Configurations")
        self.configCheck.clicked.connect(self.ConfigInfo)
        self.VBL.addWidget(self.configCheck)

        self.closeprocess = QPushButton()
        self.closeprocess.setText("Close Processes")
        self.closeprocess.clicked.connect(self.killthread)
        self.VBL.addWidget(self.closeprocess)

        self.credit = QLabel("by hello wrld, 2021.")
        self.VBL.addWidget(self.credit)

        self.Worker1 = Worker1()
        self.Worker2 = Worker2()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.AlarmUpdate.connect(self.AlarmUpdateSlot)
        self.Worker2.AlarmInterval.connect(self.AlarmTriggerSlot)
        self.setLayout(self.VBL)
        self.Worker2.start()

    def killthread(self):
        self.Worker1.terminate()
        self.Worker2.terminate()

    def AlarmTriggerSlot(self,trigger):
        if trigger == True:
            if self.comboBox.currentIndex() == 1:
                self.ExternalWindow()
            elif self.comboBox.currentIndex() == 2:
                self.SoundAlarm()
            elif self.comboBox.currentIndex() == 3:
                self.BothAlarm()

    def AlarmSelection(self):
        msg = QMessageBox()
        msg.setWindowTitle("Alarm Selection")
        msg.setIcon(QMessageBox.Information)
        msg.setText("You selected " + self.comboBox.currentText())

        x = msg.exec_()

    def BothAlarm(self):
        self.popup = PopWindow()
        global display_monitor

        monitor = QDesktopWidget().screenGeometry(display_monitor)
        self.popup.move(monitor.left(), monitor.top())
        self.popup.showFullScreen()

        self.player = QMediaPlayer()

        full_file_path = os.path.join(os.getcwd(), 'alertsound_updated.mp3')
        url = QUrl.fromLocalFile(full_file_path)
        content = QMediaContent(url)

        self.player.setMedia(content)
        self.player.play()

        msg = QMessageBox()
        msg.setWindowTitle("Sound Alarm Trigger")
        msg.setText("The sound alarm has been triggered, press OK to stop")
        msg.setIcon(QMessageBox.Warning)
        msg.buttonClicked.connect(self.muteAudioFile)

        x = msg.exec_()
    def ExternalWindow(self):
        self.popup = PopWindow()
        global display_monitor

        monitor = QDesktopWidget().screenGeometry(display_monitor)
        self.popup.move(monitor.left(), monitor.top())
        self.popup.showFullScreen()

    def CloseScreen(self):
        self.popup = PopWindow()
        self.popup.close()

    def SoundAlarm(self):
        self.player = QMediaPlayer()

        full_file_path = os.path.join(os.getcwd(), 'alertsound_updated.mp3')
        url = QUrl.fromLocalFile(full_file_path)
        content = QMediaContent(url)

        self.player.setMedia(content)
        self.player.play()

        msg = QMessageBox()
        msg.setWindowTitle("Sound Alarm Trigger")
        msg.setText("The sound alarm has been triggered, press OK to stop")
        msg.setIcon(QMessageBox.Warning)
        msg.buttonClicked.connect(self.muteAudioFile)

        x = msg.exec_()

    def muteAudioFile(self):
        self.player.setMuted(not self.player.isMuted())

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def AlarmUpdateSlot(self,violationFeed):     #alarm update; add an alarm inside the if function to activate it at the threshold.
        self.violationlabel.display(violationFeed)
        if violationFeed >= safecount:
            self.violationlabel.setStyleSheet("font: bold 20px; color: black; background-color:rgb(255, 0, 0);")
        else:
            self.violationlabel.setStyleSheet("font: bold 20px; color: black; background-color:rgb(0, 0, 255);")

    def ConfigInfo(self):
        global safecount
        global display_monitor
        global MIN_DISTANCE
        global intervalsec
        global rescale
        global USE_GPU
        restxt = ""
        cudatxt =""
        if rescale:
            restxt = "On"
        else:
            restxt = "Off"

        if USE_GPU:
            cudatxt = "On"
        else:
            cudatxt = "Off"

        msg = QMessageBox()
        msg.setWindowTitle('Configurations')
        msg.setWindowIcon(QIcon('logo.png'))
        msg.setText("Current Configurations:")
        msg.setInformativeText("Safe count: {}      ".format(safecount) +
                               "\nMinimum distance: {} px       ".format(MIN_DISTANCE) +
                               "\nMonitor display: Display {}       ".format(display_monitor+1) +
                               "\nFrame Check interval: {} second(s)       ".format(intervalsec) +
                               "\nCUDA: {}".format(cudatxt) +
                               "\nDownscaling: {}".format(restxt))

        x = msg.exec_()


class Worker2(QThread):
    AlarmInterval =pyqtSignal(int)

    def run(self):
        self.ThreadActive = True
        global globalviolate
        global safecount
        global intervalsec
        while self.ThreadActive:
            for i in range(intervalsec, 0, -1):
                time.sleep(1)
                print("interval.({})".format(i))

            frame1 = globalviolate
            print("frame 1: {}".format(frame1))
            for i in range(intervalsec, 0, -1):
                time.sleep(1)
                print("interval.({})".format(i))
            frame2 = globalviolate
            print("frame 2: {}".format(frame2))
            if frame1 and frame2 >= safecount:
                AlarmTrigger = 1
                self.AlarmInterval.emit(AlarmTrigger)
                time.sleep(3)
            else:
                frame1 = 0
                frame2 = 0
                AlarmTrigger = 0
                self.AlarmInterval.emit(AlarmTrigger)  # sends alarmtrigger to the main class



class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    AlarmUpdate = pyqtSignal(int)


    def run(self):
        global globalviolate
        global MODEL_PATH
        global USE_GPU
        global MIN_DISTANCE
        global FeedDirectory
        global rescale
        self.ThreadActive = True

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if USE_GPU:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        print("[INFO] accessing video stream...")
        vs = cv2.VideoCapture(FeedDirectory)

        def rescale480():
            vs.set(3,640)
            vs.set(4,480)
        if rescale:
            rescale480()

        def rescale_frame(frame, percent=75):
            width = int(frame.shape[1] * percent / 100)
            height = int(frame.shape[0] * percent / 100)
            dim = (width, height)
            return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)




        # loop over the frames from the video stream
        while self.ThreadActive:
            # read the next frame from the file

            (grabbed, frame) = vs.read()
            if rescale:
                frame = rescale_frame(frame, percent=30)
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break


            # resize the frame and then detect people (and only people) in it
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                                    personIdx=LABELS.index("person"))

            # initialize the set of indexes that violate the minimum social
            # distance
            violate = set()

            # ensure there are *at least* two people detections (required in
            # order to compute our pairwise distance maps)
            if len(results) >= 2:
                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number
                        # of pixels
                        if D[i, j] < MIN_DISTANCE:
                            # update our violation set with the indexes of
                            # the centroid pairs
                            violate.add(i)
                            violate.add(j)


            # loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                # if the index pair exists within the violation set, then
                # update the color
                if i in violate:
                    color = (0, 0, 255)

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)




            if grabbed:

                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0],
                                           QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                ViolationFeed = len(violate)
                globalviolate = len(violate)
                self.AlarmUpdate.emit(ViolationFeed)
                self.ImageUpdate.emit(Pic)
            else:
                break
    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = Initialization()
    Root.show()
    sys.exit(App.exec())