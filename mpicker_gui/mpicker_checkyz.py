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
                                QMenu,QMessageBox,QApplication
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


class Mpicker_checkyz(QGraphicsView):
    def __init__(self, *__args):
        super(Mpicker_checkyz, self).__init__(*__args)

    def setParameters(self, UI):
        self.UI = UI

    def checkMrcImage_yz(self):
        if self.UI.checkx <= self.UI.tomo_check.shape[2] and self.UI.checkx > 0:
            if self.UI.doubleSpinBox_xprojectLayer.value() > 0:
                mrc_image_Image = np.zeros(self.UI.tomo_check[:,:,0].shape)
                layer = int(self.UI.doubleSpinBox_xprojectLayer.value())
                checkx = self.UI.checkx - 1
                x1 = max(0, checkx - layer)
                x2 = min(self.UI.tomo_check.shape[2], checkx + layer + 1)
                if self.UI.comboBox_xProjectMode.currentText() == "Mean":
                    mrc_image_Image = self.UI.tomo_check[:,:,x1:x2].mean(axis=2)
                elif self.UI.comboBox_xProjectMode.currentText() == "Min":
                    mrc_image_Image = self.UI.tomo_check[:,:,x1:x2].min(axis=2)
                else:
                    mrc_image_Image = self.UI.tomo_check[:,:,x1:x2].max(axis=2)
            else:
                mrc_image_Image = self.UI.tomo_check[:,:,self.UI.checkx-1]
            mrc_image_Image = mrc_image_Image.transpose((1,0))
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
            self.UI.pixmap_yz = QPixmap.fromImage(qImg)
            self.UI.graphicsScene_checkyz.clear()
            self.UI.graphicsScene_checkyz.addPixmap(self.UI.pixmap_yz)
            self.UI.graphicsView_yz.setScene(self.UI.graphicsScene_checkyz)
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
            # cross.setPos(self.UI.tomo_check.shape[0] - self.UI.checkz,
            #             self.UI.tomo_check.shape[1] - self.UI.checky)
            cross.setPos(self.UI.checkz - 1, self.UI.tomo_check.shape[1] - self.UI.checky)
            self.UI.graphicsScene_checkyz.addItem(cross)


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.UI.pixmap_yz is not None:
            self.UI.pressZ = int(round(self.mapToScene(event.pos()).x()+1))
            #self.UI.pressZ = self.UI.tomo_check.shape[0] - round(self.mapToScene(event.pos()).x())
            self.UI.pressY = int(self.UI.tomo_check.shape[1] - round(self.mapToScene(event.pos()).y()))
            if event.button() & Qt.LeftButton:
                if self.UI.pressZ > 0 and self.UI.pressZ <= self.UI.tomo_check.shape[0] and \
                        self.UI.pressY > 0 and self.UI.pressY <= self.UI.tomo_check.shape[1]:
                    self.UI.checkz = self.UI.pressZ
                    self.UI.checky = self.UI.pressY
                    self.checkMrcImage_yz()

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
        self.UI.graphicsView_xz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_xz.scale(1.1, 1.1)

    def zoomOut(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1/1.1,1/1.1)
        self.UI.graphicsView_xy.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_xy.scale(1/1.1, 1/1.1)
        self.UI.graphicsView_xz.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_xz.scale(1 / 1.1, 1 / 1.1)