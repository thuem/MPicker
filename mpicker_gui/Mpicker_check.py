#!/usr/bin/env python3

# Copyright (C) 2024  Xiaofeng Yan, Shudong Li
# Xueming Li Lab, Tsinghua University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from PyQt5.QtWidgets import     QMainWindow,QFileDialog,QGraphicsView,QGraphicsScene,\
                                QMenu,QMessageBox,QApplication,\
                                QVBoxLayout,QWidget,QLabel,QFrame,\
                                QShortcut
from PyQt5.QtCore import QThread,pyqtSignal,Qt,QRectF
from PyQt5.QtGui import QPixmap,QImage,QCursor,QFont,QKeySequence,QTransform
from PyQt5 import uic,QtCore
import os,sys
import numpy as np
import mrcfile
import math
from mpicker_item import Cross,Circle
from mpicker_checkxy import Mpicker_checkxy

# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS

    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Mpicker_check(QMainWindow):
    def __init__(self):
        super(Mpicker_check, self).__init__()
        # Load the ui file
        # self.uifile_path = resource_path("check.ui")
        self.uifile_path    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.ui")
        uic.loadUi(self.uifile_path, self)
        self.show_raw_flag  = True
        #self.show()

    def show(self) -> None:
        super().show()
        self.init_splitter()

    def init_splitter(self):
        self.splitter1.splitterMoved.connect(self.slot_splitter1)
        self.splitter2.splitterMoved.connect(self.slot_splitter2)
        pos=self.splitter1.handle(1).x()
        self.splitter2.moveSplitter(pos,1)
        self.splitter1.moveSplitter(pos,1)

    def slot_splitter1(self,pos,idx):
        if self.splitter1.press_flag: self.splitter2.moveSplitter(pos,idx)

    def slot_splitter2(self,pos,idx):
        if self.splitter2.press_flag: self.splitter1.moveSplitter(pos,idx)

    def setParameters(self, UI):
        self.UI             = UI
        self.checkx         = 1
        self.checky         = 1
        self.checkz         = 1
        self.tomo_check     = None
        self.pixmap_xy      = None
        self.pixmap_xz      = None
        self.pixmap_yz      = None
        self.pressX         = None
        self.pressY         = None
        self.check_min      = 0
        self.check_max      = 255
        self.new_min        = 0
        self.new_max        = 255
        self.check_contrast = 0
        self.check_std      = 0
        self.check_mean     = 0
        self.project_flag   = False
        self.check_Contrast_Max    = 2550
        self.check_Bright_Max      = 2550
        self.check_Contrast_value   = 0
        self.check_Bright_value     = 0
        self.check_tomo_max         = 0
        self.check_tomo_min         = 0

        self.graphicsScene_checkxy = QGraphicsScene()
        self.graphicsView_xy.setParameters(self)
        self.graphicsView_xy.setDragMode(self.graphicsView_xy.ScrollHandDrag)
        self.graphicsView_xy.setTransformationAnchor(self.graphicsView_xy.AnchorUnderMouse)

        self.graphicsScene_checkxz = QGraphicsScene()
        self.graphicsView_xz.setParameters(self)
        self.graphicsView_xz.setDragMode(self.graphicsView_xz.ScrollHandDrag)
        self.graphicsView_xz.setTransformationAnchor(self.graphicsView_xz.AnchorUnderMouse)

        self.graphicsScene_checkyz = QGraphicsScene()
        self.graphicsView_yz.setParameters(self)
        self.graphicsView_yz.setDragMode(self.graphicsView_yz.ScrollHandDrag)
        self.graphicsView_yz.setTransformationAnchor(self.graphicsView_yz.AnchorUnderMouse)

        self.horizontalSlider_z.valueChanged.connect(self.slide_z)
        self.horizontalSlider_x.valueChanged.connect(self.slide_x)
        self.horizontalSlider_y.valueChanged.connect(self.slide_y)

        self.doubleSpinBox_X.valueChanged.connect(self.SpinBox_X)
        self.doubleSpinBox_Y.valueChanged.connect(self.SpinBox_Y)
        self.doubleSpinBox_Z.valueChanged.connect(self.SpinBox_Z)

        self.pushButton_Goto.clicked.connect(self.connect_coord)
        self.pushButton_Screenshot.clicked.connect(self.save_pixmap)

        self.horizontalSlider_checkBright.valueChanged.connect(self.slide_Bright)
        self.horizontalSlider_checkContrast.valueChanged.connect(self.slide_Contrast)
        self.doubleSpinBox_checkBright.valueChanged.connect(self.doubleSpinBox_check_Bright)
        self.doubleSpinBox_checkContrast.valueChanged.connect(self.doubleSpinBox_check_Contrast)
        self.doubleSpinBox_zprojectLayer.valueChanged.connect(self.doubleSpinBox_check_projectLayer)
        self.comboBox_zProjectMode.activated.connect(self.change_Projectmode)
        self.doubleSpinBox_yprojectLayer.valueChanged.connect(self.doubleSpinBox_check_projectLayer)
        self.comboBox_yProjectMode.activated.connect(self.change_Projectmode)
        self.doubleSpinBox_xprojectLayer.valueChanged.connect(self.doubleSpinBox_check_projectLayer)
        self.comboBox_xProjectMode.activated.connect(self.change_Projectmode)
        self.Init_show_mrc()

    def Init_show_mrc(self):
        if self.UI.show_mrc is not None:
            with mrcfile.open(self.UI.show_mrc) as tomo:
                self.tomo_check = tomo.data[:,::-1,:]
                #self.tomo_check = np.flip(self.tomo_check,axis = 0)
                # tomo_max = np.max(self.tomo_check)
                # tomo_min = np.min(self.tomo_check)
                # self.tomo_check = ((self.tomo_check - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
            self.checkz = int(round(1 / 2 * self.tomo_check.shape[0]))
            self.checky = int(round(1 / 2 * self.tomo_check.shape[1]))
            self.checkx = int(round(1 / 2 * self.tomo_check.shape[2]))

            tomo_max    = np.max(self.tomo_check).astype(float)
            tomo_min    = np.min(self.tomo_check).astype(float)
            tomo_all    = tomo_max - tomo_min

            if self.show_raw_flag is False:
                if tomo_max != tomo_min:
                    self.tomo_check = ((self.tomo_check - tomo_min) / (tomo_max - tomo_min) * 255).astype(
                        np.uint8)
                else:
                    self.tomo_check = np.zeros(self.tomo_check.shape)

            self.check_min  = np.min(self.tomo_check)
            self.check_max  = np.max(self.tomo_check)
            self.check_std  = float(np.std(self.tomo_check)) # round(np.std(self.tomo_check),1)
            if self.check_std == 0:
                self.check_std = 0.1
            self.check_mean = np.average(self.tomo_check)

            # reset the origin
            if np.std(self.tomo_check) != 0:
                self.check_Contrast_Max     = round(math.ceil(tomo_all / np.std(self.tomo_check)) * 10,1)
            else:
                self.check_Contrast_Max     = 0
            self.check_Bright_Max       = round(self.check_Contrast_Max / 2)
            self.check_Contrast_value   = round((self.check_Contrast_Max /2 /10))
            self.check_Bright_value     = 0 # np.average
            self.check_tomo_min         = self.check_mean + (self.check_Bright_value - self.check_Contrast_value) * self.check_std
            self.check_tomo_max         = self.check_mean + (self.check_Bright_value + self.check_Contrast_value) * self.check_std
            self.check_contrast         = 255 / (self.check_tomo_max - self.check_tomo_min)

            # reset the slider_contrast
            self.doubleSpinBox_checkBright.setMinimum(-self.check_Bright_Max/10)
            self.doubleSpinBox_checkBright.setMaximum(self.check_Bright_Max/10)
            self.doubleSpinBox_checkContrast.setMinimum(0.1)
            self.doubleSpinBox_checkContrast.setMaximum(self.check_Contrast_Max/10)
            self.horizontalSlider_checkContrast.setMinimum(1)
            self.horizontalSlider_checkContrast.setMaximum(self.check_Contrast_Max)
            self.horizontalSlider_checkContrast.setValue(self.check_Contrast_value * 10)
            self.horizontalSlider_checkBright.setMinimum(-self.check_Bright_Max)
            self.horizontalSlider_checkBright.setMaximum(self.check_Bright_Max)
            self.horizontalSlider_checkBright.setValue(self.check_Bright_value * 10)

            self.label_contrastlimit.setText(f"/{self.check_Contrast_Max/10}")
            self.label_brightlimit.setText(f"/{self.check_Bright_Max/10}")

            self.horizontalSlider_z.setMinimum(1)
            self.horizontalSlider_z.setMaximum(self.tomo_check.shape[0])
            self.horizontalSlider_z.setValue(self.checkz)
            self.horizontalSlider_x.setMinimum(1)
            self.horizontalSlider_x.setMaximum(self.tomo_check.shape[2])
            self.horizontalSlider_x.setValue(self.checkx)
            self.horizontalSlider_y.setMinimum(1)
            self.horizontalSlider_y.setMaximum(self.tomo_check.shape[1])
            self.horizontalSlider_y.setValue(self.checky)

            self.doubleSpinBox_X.setMinimum(1)
            self.doubleSpinBox_X.setMaximum(self.tomo_check.shape[2])
            self.doubleSpinBox_Y.setMinimum(1)
            self.doubleSpinBox_Y.setMaximum(self.tomo_check.shape[1])
            self.doubleSpinBox_Z.setMinimum(1)
            self.doubleSpinBox_Z.setMaximum(self.tomo_check.shape[0])

            self.doubleSpinBox_X.setValue(self.checkx)
            self.doubleSpinBox_Y.setValue(self.checky)
            self.doubleSpinBox_Z.setValue(self.checkz)

            self.x_pos_label.setText(f"/{self.tomo_check.shape[2]}")
            self.y_pos_label.setText(f"/{self.tomo_check.shape[1]}")
            self.z_pos_label.setText(f"/{self.tomo_check.shape[0]}")

            self.label_zprojectLayermax.setText(f"/{math.floor(self.tomo_check.shape[0]/2)}")
            self.doubleSpinBox_zprojectLayer.setMaximum(math.floor(self.tomo_check.shape[0]/2))
            self.label_yprojectLayermax.setText(f"/{math.floor(self.tomo_check.shape[1] / 2)}")
            self.doubleSpinBox_yprojectLayer.setMaximum(math.floor(self.tomo_check.shape[1] / 2))
            self.label_xprojectLayermax.setText(f"/{math.floor(self.tomo_check.shape[2] / 2)}")
            self.doubleSpinBox_xprojectLayer.setMaximum(math.floor(self.tomo_check.shape[2] / 2))

            self.graphicsView_xy.checkMrcImage()



    def slide_z(self,value):
        if self.pixmap_xy is not None:
            self.checkz = value
            self.graphicsView_xy.checkMrcImage()

    def slide_x(self,value):
        if self.pixmap_xy is not None:
            self.checkx = value
            self.graphicsView_xy.checkMrcImage()

    def slide_y(self,value):
        if self.pixmap_xy is not None:
            self.checky = value
            self.graphicsView_xy.checkMrcImage()

    def SpinBox_X(self):
        if self.pixmap_xy is not None:
            self.horizontalSlider_x.setValue(int(self.doubleSpinBox_X.value()))
            self.graphicsView_xy.checkMrcImage()

    def SpinBox_Y(self,value):
        if self.pixmap_xy is not None:
            self.horizontalSlider_y.setValue(int(self.doubleSpinBox_Y.value()))
            self.graphicsView_xy.checkMrcImage()

    def SpinBox_Z(self,value):
        if self.pixmap_xy is not None:
            self.horizontalSlider_z.setValue(int(self.doubleSpinBox_Z.value()))
            self.graphicsView_xy.checkMrcImage()

    def slide_Contrast(self, value):
        if self.pixmap_xy is not None and self.show_raw_flag:
            self.check_Contrast_value  = round(value / 10,1)
            self.check_tomo_min = self.check_mean + (
                        self.check_Bright_value - self.check_Contrast_value) * self.check_std
            self.check_tomo_max = self.check_mean + (
                        self.check_Bright_value + self.check_Contrast_value) * self.check_std
            self.check_contrast = 255 / (self.check_tomo_max - self.check_tomo_min)
            # self.check_contrast         = (self.check_tomo_max - self.check_tomo_min) / (
            #                                 self.check_max - self.check_min)
            self.graphicsView_xy.checkMrcImage()

    def slide_Bright(self, value):
        if self.pixmap_xy is not None and self.show_raw_flag:
            self.check_Bright_value    = round(value / 10,1)
            self.check_tomo_min = self.check_mean + (
                        self.check_Bright_value - self.check_Contrast_value) * self.check_std
            self.check_tomo_max = self.check_mean + (
                        self.check_Bright_value + self.check_Contrast_value) * self.check_std
            self.check_contrast = 255 / (self.check_tomo_max - self.check_tomo_min)
            # self.check_contrast         = (self.check_tomo_max - self.check_tomo_min) / (
            #                                 self.check_max - self.check_min)
            self.graphicsView_xy.checkMrcImage()

    def doubleSpinBox_check_Contrast(self):
        if self.pixmap_xy is not None and self.show_raw_flag:
            self.check_Contrast_value = self.doubleSpinBox_checkContrast.value()
            self.horizontalSlider_checkContrast.setValue(int(self.check_Contrast_value * 10))
            self.graphicsView_xy.checkMrcImage()
        else:
            pass

    def doubleSpinBox_check_Bright(self):
        if self.pixmap_xy is not None and self.show_raw_flag:
            self.check_Bright_value = self.doubleSpinBox_checkBright.value()
            self.horizontalSlider_checkBright.setValue(int(self.check_Bright_value * 10))
            self.graphicsView_xy.checkMrcImage()
        else:
            pass

    def doubleSpinBox_check_projectLayer(self):
        if self.pixmap_xy is not None:
            self.graphicsView_xy.checkMrcImage()

    def change_Projectmode(self):
        if self.pixmap_xy is not None:
            self.graphicsView_xy.checkMrcImage()

    def connect_coord(self):
        if self.UI.tomo_result is None:
            return
        if self.UI.tomo_result.shape == self.tomo_check.shape:
            # copy from SpinBox_resultX
            self.UI.resultx = self.checkx
            self.UI.resulty = self.checky
            self.UI.resultz = self.checkz
            # self.UI.label_Press_Flag = False
            self.UI.showResultImage()
            self.UI.activateWindow()

    def save_pixmap(self):
        if self.pixmap_xy is None or self.pixmap_yz is None or self.pixmap_xz is None:
            return
        idx = 0
        while True:
            fnamexy = os.path.abspath(f"XYZ_xy_{idx:03}.tif")
            if os.path.exists(fnamexy):
                idx += 1
                continue
            fnamezy = os.path.abspath(f"XYZ_zy_{idx:03}.tif")
            if os.path.exists(fnamezy):
                idx += 1
                continue
            fnamexz = os.path.abspath(f"XYZ_xz_{idx:03}.tif")
            if os.path.exists(fnamexz):
                idx += 1
                continue
            break
        self.pixmap_xy.save(fnamexy, "TIFF")
        self.pixmap_yz.save(fnamezy, "TIFF")
        self.pixmap_xz.save(fnamexz, "TIFF")
        print(f"save images as:\n{fnamexy}\n{fnamezy}\n{fnamexz}")
        return
    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: Mpicker_check.py input.mrc")
        exit()
    fin = sys.argv[1]
    if os.path.exists(fin) is False:
        print(f"File {fin} does not exist.")
        print("Usage: Mpicker_check.py input.mrc")
        exit()
    class fakeUI:
        def __init__(self):
            self.show_mrc = fin
            self.tomo_result = None
    app = QApplication(sys.argv)
    window = Mpicker_check()
    window.setParameters(fakeUI())
    window.show()
    sys.exit(app.exec_())