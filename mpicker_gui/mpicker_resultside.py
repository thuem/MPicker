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

from PyQt5.QtWidgets import     QGraphicsView,QApplication,\
                                QMenu,QAction,\
                                QShortcut,QLabel,QMessageBox
from PyQt5.QtCore import Qt
from PIL import Image,ImageDraw,ImageEnhance
import numpy as np
from PyQt5.QtGui import QPixmap,QImage,QCursor,QFont,QKeySequence
import re
from mpicker_item import Cross,LongCross

class Mpicker_ResultSide(QGraphicsView):
    def __init__(self, *__args):
        super(Mpicker_ResultSide, self).__init__(*__args)

    def setParameters(self, UI):
        self.UI = UI
        self.UI.showResultImage_side = self.showResultImage_side
        self.UI.removeResultCursor_side = self.removeResultCursor_side
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.GraphicViewResultMenu)
        # # Gap and line width setting
        # self.length = 5
        # self.width = 2

    def GraphicViewResultMenu(self):
        if self.UI.sidepixmap is not None:
            # initial the Menu
            self.groupBox_menu = QMenu(self)
            self.actionA = QAction(u'Save screenshot', self)
            self.actionA.triggered.connect(lambda : self.UI.save_pixmap(self.UI.screenshot_Fyz, "Flatten_zy"))
            self.groupBox_menu.addAction(self.actionA)      
            self.groupBox_menu.popup(QCursor.pos())

    def showResultImage_side(self, hidecross=False):
        if not self.UI.allow_showResultImage_side:
            return
        mrc_image_Image = self.UI.tomo_result[:, :, int(round(self.UI.resultx - 1))].transpose((1, 0))
        #mrc_image_Image = self.UI.result_tomo_min + self.UI.result_contrast * (mrc_image_Image - self.UI.old_result_tomo_min)
        mrc_image_Image = np.select(
            [(mrc_image_Image > self.UI.result_tomo_min) & (mrc_image_Image < self.UI.result_tomo_max),
             mrc_image_Image <= self.UI.result_tomo_min,
             mrc_image_Image >= self.UI.result_tomo_max],
            [self.UI.result_contrast * (mrc_image_Image - self.UI.result_tomo_min),
             0,
             255])
        mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
        mrc_image_Image = np.asarray(mrc_image_Image)
        # Show the Image
        height, width, channels = mrc_image_Image.shape
        bytesPerLine = channels * width
        qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.UI.sidepixmap = QPixmap.fromImage(qImg)
        self.UI.graphicsScene_resultside.clear()
        self.UI.graphicsScene_resultside.addPixmap(self.UI.sidepixmap)
        self.UI.screenshot_Fyz = self.UI.sidepixmap
        self.UI.graphicsView_resultside.setScene(self.UI.graphicsScene_resultside)
        longcross = LongCross()
        longcross.setPos(self.UI.resultz - 1, self.UI.tomo_result.shape[1] - self.UI.resulty)
        # if self.UI.label_Press_Flag == False:
            # longcross.setColor("red")
        # else:
        #     longcross.setColor("gray")
        if hidecross:
            longcross.setColor("gray")
        else:
            longcross.setColor("red")
        self.UI.graphicsScene_resultside.addItem(longcross)

    def removeResultCursor_side(self):
        self.UI.graphicsScene_resultside.clear()
        self.UI.graphicsScene_resultside.addPixmap(self.UI.sidepixmap)
        self.UI.screenshot_Fyz = self.UI.sidepixmap
        self.UI.graphicsView_resultside.setScene(self.UI.graphicsScene_resultside)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.UI.resultpixmap is not None:
            self.pressZ = int(round(self.mapToScene(event.pos()).x() + 1))
            self.pressY = int(self.UI.tomo_result.shape[1] - round(self.mapToScene(event.pos()).y()))

            if event.button() & Qt.LeftButton:
                if self.pressZ > 0 and self.pressZ < self.UI.tomo_result.shape[0] and self.pressY > 0 and self.pressY < self.UI.tomo_result.shape[1]:
                    #coord y is reversed in 3Dmod!
                    self.UI.resultz = self.pressZ
                    self.UI.resulty = self.pressY
                    # self.UI.label_Press_Flag = False
                    self.UI.showResultImage()

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
        self.scale(1.1, 1.1)
        self.UI.graphicsView_result.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_result.scale(1.1, 1.1)
        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # self.scale(1.1,1.1)

    def zoomOut(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1 / 1.1, 1 / 1.1)
        self.UI.graphicsView_result.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_result.scale(1 / 1.1, 1 / 1.1)
        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # self.scale(1/1.1,1/1.1)