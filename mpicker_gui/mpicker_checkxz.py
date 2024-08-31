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
from PIL import Image,ImageEnhance
import os,sys
import argparse
import numpy as np
import mrcfile
import copy
from mpicker_item import Cross,Circle


class Mpicker_checkxz(QGraphicsView):
    def __init__(self, *__args):
        super(Mpicker_checkxz, self).__init__(*__args)

    def setParameters(self, UI):
        self.UI = UI

    def checkMrcImage_xz(self):
        if self.UI.checky <= self.UI.tomo_check.shape[1] and self.UI.checky > 0:
            if self.UI.doubleSpinBox_yprojectLayer.value() > 0:
                mrc_image_Image = np.zeros(self.UI.tomo_check[:,0,:].shape)
                layer = int(self.UI.doubleSpinBox_yprojectLayer.value())
                checky = self.UI.tomo_check.shape[1] - self.UI.checky
                y1 = max(0, checky - layer)
                y2 = min(self.UI.tomo_check.shape[1], checky + layer + 1)
                if self.UI.comboBox_yProjectMode.currentText() == "Mean":
                    mrc_image_Image = self.UI.tomo_check[:,y1:y2,:].mean(axis=1)
                elif self.UI.comboBox_yProjectMode.currentText() == "Min":
                    mrc_image_Image = self.UI.tomo_check[:,y1:y2,:].min(axis=1)
                else:
                    mrc_image_Image = self.UI.tomo_check[:,y1:y2,:].max(axis=1)
            else:
                mrc_image_Image = self.UI.tomo_check[:,self.UI.tomo_check.shape[1] - self.UI.checky,:]
            #mrc_image_Image = self.UI.check_tomo_min + self.UI.check_contrast * (mrc_image_Image - self.UI.check_min)
            if self.UI.show_raw_flag:
                mrc_image_Image = np.select(
                    [(mrc_image_Image > self.UI.check_tomo_min) & (mrc_image_Image < self.UI.check_tomo_max),
                     mrc_image_Image <= self.UI.check_tomo_min,
                     mrc_image_Image >= self.UI.check_tomo_max],
                    [self.UI.check_contrast * (mrc_image_Image - self.UI.check_tomo_min),
                     0,
                     255])
            mrc_image_Image = np.flip(mrc_image_Image,axis = 0)
            mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
            mrc_image_Image = np.asarray(mrc_image_Image)
            height, width, channels = mrc_image_Image.shape
            bytesPerLine = channels * width
            qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.UI.pixmap_xz = QPixmap.fromImage(qImg)
            self.UI.graphicsScene_checkxz.clear()
            self.UI.graphicsScene_checkxz.addPixmap(self.UI.pixmap_xz)
            self.UI.graphicsView_xz.setScene(self.UI.graphicsScene_checkxz)
            self.refresh_Cursor()
            self.UI.horizontalSlider_z.setValue(int(self.UI.checkz))
            self.UI.horizontalSlider_x.setValue(int(self.UI.checkx))
            self.UI.horizontalSlider_y.setValue(int(self.UI.checky))
            self.UI.doubleSpinBox_X.setValue(self.UI.checkx)
            self.UI.doubleSpinBox_Y.setValue(self.UI.checky)
            self.UI.doubleSpinBox_Z.setValue(self.UI.checkz)
            self.UI.doubleSpinBox_checkContrast.setValue(self.UI.check_Contrast_value)
            self.UI.doubleSpinBox_checkBright.setValue(self.UI.check_Bright_value)

    def refresh_Cursor(self):
        if self.UI.pixmap_xz is not None:
            cross = Cross()
            cross.setPos(self.UI.checkx - 1, self.UI.tomo_check.shape[0] - self.UI.checkz)
            # cross.setPos(self.UI.checkx - 1, self.UI.checkz - 1)
            self.UI.graphicsScene_checkxz.addItem(cross)


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.UI.pixmap_xz is not None:
            self.UI.pressX = int(round(self.mapToScene(event.pos()).x()+1))
            self.UI.pressZ = int(self.UI.tomo_check.shape[0] - round(self.mapToScene(event.pos()).y()))
            #self.UI.pressZ = round(self.mapToScene(event.pos()).y()+1)
            if event.button() & Qt.LeftButton:
                if self.UI.pressX > 0 and self.UI.pressX <= self.UI.tomo_check.shape[2] and \
                        self.UI.pressZ > 0 and self.UI.pressZ <= self.UI.tomo_check.shape[0]:
                    # coord y is reversed in 3Dmod!
                    self.UI.checkx = self.UI.pressX
                    self.UI.checkz = self.UI.pressZ
                    self.checkMrcImage_xz()

    def enterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CrossCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoomIn()
        else:
            self.zoomOut()

    def zoomIn(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1.1,1.1)
        self.UI.graphicsView_xy.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_xy.scale(1.1, 1.1)
        self.UI.graphicsView_yz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_yz.scale(1.1, 1.1)

    def zoomOut(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1/1.1,1/1.1)
        self.UI.graphicsView_xy.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_xy.scale(1/1.1, 1/1.1)
        self.UI.graphicsView_yz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_yz.scale(1/1.1, 1/1.1)
