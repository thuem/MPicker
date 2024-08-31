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


class Mpicker_checkxy(QGraphicsView):
    def __init__(self, *__args):
        super(Mpicker_checkxy, self).__init__(*__args)

    def setParameters(self, UI):
        self.UI = UI

    def checkMrcImage(self):
        if self.UI.checkz <= self.UI.tomo_check.shape[0] and self.UI.checkz > 0:
            if self.UI.doubleSpinBox_zprojectLayer.value() > 0:
                mrc_image_Image = np.zeros(self.UI.tomo_check[0,:,:].shape)
                layer = int(self.UI.doubleSpinBox_zprojectLayer.value())
                checkz = self.UI.checkz - 1
                z1 = max(0, checkz - layer)
                z2 = min(self.UI.tomo_check.shape[0], checkz + layer + 1)
                #z1, z2 = self.UI.tomo_check.shape[0] - 1 - z2, self.UI.tomo_check.shape[0] - 1 - z1
                if self.UI.comboBox_zProjectMode.currentText() == "Mean":
                    mrc_image_Image = self.UI.tomo_check[z1:z2,:,:].mean(axis=0)
                elif self.UI.comboBox_zProjectMode.currentText() == "Min":
                    mrc_image_Image = self.UI.tomo_check[z1:z2,:,:].min(axis=0)
                else:
                    mrc_image_Image = self.UI.tomo_check[z1:z2,:,:].max(axis=0)
            else:
                #mrc_image_Image = self.UI.tomo_check[self.UI.tomo_check.shape[0] - self.UI.checkz,: , :]
                mrc_image_Image = self.UI.tomo_check[self.UI.checkz - 1, :, :]
            #mrc_image_Image = self.UI.check_tomo_min + self.UI.check_contrast * (mrc_image_Image - self.UI.check_min)
            if self.UI.show_raw_flag:
                mrc_image_Image = np.select(
                    [(mrc_image_Image > self.UI.check_tomo_min) & (mrc_image_Image < self.UI.check_tomo_max),
                     mrc_image_Image <= self.UI.check_tomo_min,
                     mrc_image_Image >= self.UI.check_tomo_max],
                    [self.UI.check_contrast * (mrc_image_Image - self.UI.check_tomo_min),
                     0,
                     255])
            mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
            mrc_image_Image = np.asarray(mrc_image_Image)
            height, width, channels = mrc_image_Image.shape
            bytesPerLine = channels * width
            qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.UI.pixmap_xy = QPixmap.fromImage(qImg)
            self.UI.graphicsScene_checkxy.clear()
            self.UI.graphicsScene_checkxy.addPixmap(self.UI.pixmap_xy)
            self.UI.graphicsView_xy.setScene(self.UI.graphicsScene_checkxy)
            self.refresh_Cursor()
            self.UI.graphicsView_xz.checkMrcImage_xz()
            self.UI.graphicsView_yz.checkMrcImage_yz()
            self.UI.horizontalSlider_z.setValue(int(self.UI.checkz))
            self.UI.horizontalSlider_x.setValue(int(self.UI.checkx))
            self.UI.horizontalSlider_y.setValue(int(self.UI.checky))
            self.UI.doubleSpinBox_X.setValue(self.UI.checkx)
            self.UI.doubleSpinBox_Y.setValue(self.UI.checky)
            self.UI.doubleSpinBox_Z.setValue(self.UI.checkz)
            self.UI.doubleSpinBox_checkContrast.setValue(self.UI.check_Contrast_value)
            self.UI.doubleSpinBox_checkBright.setValue(self.UI.check_Bright_value)
            #print("self.UI.checkz = ",self.UI.checkz)

    def refresh_Cursor(self):
        if self.UI.pixmap_xy is not None:
            cross = Cross()
            cross.setPos(self.UI.checkx - 1, self.UI.tomo_check.shape[1] - self.UI.checky)
            self.UI.graphicsScene_checkxy.addItem(cross)


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.UI.pixmap_xy is not None:
            self.UI.pressX = int(round(self.mapToScene(event.pos()).x()+1))
            self.UI.pressY = int(self.UI.tomo_check.shape[1] - round(self.mapToScene(event.pos()).y()))
            if event.button() & Qt.LeftButton:
                if self.UI.pressX > 0 and self.UI.pressX <= self.UI.tomo_check.shape[2] and \
                        self.UI.pressY > 0 and self.UI.pressY <= self.UI.tomo_check.shape[1]:
                    self.UI.checkx = self.UI.pressX
                    self.UI.checky = self.UI.pressY
                    self.checkMrcImage()

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
        self.UI.graphicsView_xz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_xz.scale(1.1,1.1)
        self.UI.graphicsView_yz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_yz.scale(1.1, 1.1)

    def zoomOut(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1/1.1,1/1.1)
        self.UI.graphicsView_xz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_xz.scale(1/1.1,1/1.1)
        self.UI.graphicsView_yz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_yz.scale(1/1.1,1/1.1)


