
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

from PyQt5.QtWidgets import  *
from PyQt5.QtCore import Qt,QRectF
from PyQt5.QtGui import QPixmap,QImage,QCursor,QFont,QKeySequence,QPainter,QPen,QTransform,QColor
import numpy as np
from typing import Union

class Cross(QGraphicsItem):
    def __init__(self, *__args, up=True):
        super(Cross, self).__init__(*__args)
        self.setFlag(self.ItemIgnoresTransformations)
        self.color  = Qt.red
        self.up = up # normal
    def boundingRect(self):
        penWidth = 1.0
        return QRectF(-15 - penWidth / 2, -15 - penWidth / 2,
                      30 + penWidth, 30 + penWidth)

    def paint(self, painter, option, widget):
        if self.color == Qt.darkGray:
            return
        painter.setPen(self.color)
        if self.up:
            painter.drawLine(-6,0,6,0)
            painter.drawLine(0,-6,0,6)
        else:
            painter.drawLine(-5,-5,5,5)
            painter.drawLine(-5,5,5,-5)
        # painter.drawEllipse(100,100,20,20)
        # painter.drawRoundedRect(20, 20, 40, 30, 10, 10) #(x,y,w,h)
    def setColor(self, str : Union[str,QColor]):
        if type(str) is QColor:
            self.color = str
        elif str == "gray":
            # color = QColor(250, 250, 200)#QColor
            self.color = Qt.darkGray
        elif str == "red" or str == "different":
            self.color = Qt.red
        elif str == "green":
            self.color = Qt.green
        elif str == "blue":
            self.color = Qt.blue
        # elif str == "magenta" or str == "different":
        #     self.color = Qt.magenta
        else:
            print("no color")

class LongCross(QGraphicsItem):
    def __init__(self, *__args):
        super(LongCross, self).__init__(*__args)
        self.setFlag(self.ItemIgnoresTransformations)
        self.color  = Qt.red
        self.len    = 20

    def boundingRect(self):
        penWidth = 1.0
        return QRectF(-5 - penWidth / 2, -self.len/2 - penWidth / 2,
                      10 + penWidth, self.len + penWidth)

    def paint(self, painter, option, widget):
        if self.color == Qt.darkGray:
            return
        painter.setPen(self.color)
        painter.drawLine(-5,0,5,0)
        painter.drawLine(0,-self.len,0,self.len)
        # painter.drawEllipse(100,100,20,20)
        # painter.drawRoundedRect(20, 20, 40, 30, 10, 10) #(x,y,w,h)
    def setColor(self,str):
        if str == "gray":
            # color = QColor(250, 250, 200)#QColor
            self.color = Qt.darkGray
        elif str == "red":
            self.color = Qt.red
        elif str == "blue":
            self.color = Qt.blue
        else:
            print("no color")
    def setlen(self,int):
        self.len = int


class Circle(QGraphicsItem):
    def __init__(self, *__args):
        super(Circle, self).__init__(*__args)
        # self.setFlag(self.ItemIgnoresTransformations)
        self.color  = Qt.blue
        self.radius = 5

    def boundingRect(self):
        penWidth = 1.0
        return QRectF(-self.radius - penWidth / 2, -self.radius - penWidth / 2,
                      2*self.radius + penWidth, 2*self.radius + penWidth)

    def paint(self, painter, option, widget):
        painter.setPen(self.color)
        painter.drawEllipse(-self.radius,-self.radius,2*self.radius,2*self.radius)
        # painter.drawEllipse(100,100,20,20)
        # painter.drawRoundedRect(20, 20, 40, 30, 10, 10) #(x,y,w,h)

    def setR(self,input_radius):
        self.radius = input_radius

    def setRZ(self, radius, dz):
        if dz >= radius:
            return
        if radius < 10:
            self.radius = radius - dz
        else:
            self.radius = int(round(np.sqrt(radius**2 - dz**2)))

    def setColor(self, str : Union[str,QColor]):
        if type(str) is QColor:
            self.color = str
        elif str == "gray":
            color = QColor(250, 250, 200)  # QColor
            self.color = Qt.darkGray
        elif str == "red" or str == "different":
            self.color = Qt.red
        elif str == "green":
            self.color = Qt.green
        elif str == "blue":
            self.color = Qt.blue
        # elif str == "magenta" or str == "different":
        #     self.color = Qt.magenta
        else:
            print("no color")


class Line(QGraphicsLineItem):
    def __init__(self, *__args):
        super(Line, self).__init__(*__args)
        self.color = Qt.blue
        self.setPen(self.color)
    def setColor(self, str : Union[str,QColor]):
        if type(str) is QColor:
            self.color = str
        elif str == "blue":
            self.color = Qt.blue
        elif str == "red" or str == "different":
            self.color = Qt.red
        self.setPen(self.color)


class Arrow(QGraphicsItem):
    def __init__(self, *__args):
        super(Arrow, self).__init__(*__args)
        #self.setFlag(self.ItemIgnoresTransformations)
        self.color      = Qt.green

    def boundingRect(self):
        penWidth = 1.0
        return QRectF(-22 - penWidth / 2, -7 - penWidth / 2,
                      17 + penWidth, 12 + penWidth)

    def paint(self, painter, option, widget):
        painter.setPen(self.color)
        painter.drawLine(-17,  0, 0, 0)
        painter.drawLine( -7, -7, 0, 0)
        painter.drawLine( -7,  7, 0, 0)


def give_colorbar(mode, w=70, s=40, maxv=3): ##change s50
    image=np.ones((256+2*s,w+s//2+5,3),dtype=np.uint8)*255
    space=max(0,(w-50)//2)
    if mode == 'area':
        for i in range(256):
            if i<128:
                image[i+s,space+5:space+50+5,0]=255
                image[i+s,space+5:space+50+5,1:3]=i*2
            else:
                image[i+s,space+5:space+50+5,2]=255
                image[i+s,space+5:space+50+5,0:2]=255-(i-128)*2
    elif mode == 'stretch':
        for i in range(256):
                image[i+s,space+5:space+50+5,0]=255
                image[i+s,space+5:space+50+5,1:3]=i
    height, width, channels = image.shape
    bytesPerLine = channels * width
    qimage=QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

    painter=QPainter(qimage) # must paint after QApplication if drawText ??
    pen = QPen()
    pen.setWidth(2)
    painter.setPen(pen)
    painter.drawRect(space+5-1,s-1,50+1,256+1)

    font1 = QFont()
    font1.setFamily('Agency FB')
    font1.setPixelSize(s//2)
    painter.setFont(font1)
    if mode == 'area':
        painter.drawText(5, 0,w,s,Qt.AlignHCenter|Qt.AlignTop, 'contract')
        painter.drawText(5, 256+s,w,s,Qt.AlignHCenter|Qt.AlignBottom, 'expand')
    elif mode == 'stretch':
        painter.drawText(5, 0,w,s,Qt.AlignHCenter|Qt.AlignTop, 'stretch')

    font2 = QFont()
    font2.setFamily('Agency FB')
    font2.setPixelSize(s//3)
    painter.setFont(font2)
    if mode == 'area':
        painter.drawText(space+50+5+10, 128+s//2,s,s,Qt.AlignVCenter, '1')
        for i in range(maxv):
            dy=int(round(128/(maxv-1)*i))
            v=str(i+1)
            painter.drawText(space+50+5+10, 128+s//2+dy,s,s,Qt.AlignVCenter, v)
            painter.drawText(space+50+5+10, 128+s//2-dy,s,s,Qt.AlignVCenter, v)
    elif mode == 'stretch':
        for i in range(maxv):
            dy=int(round(256/(maxv-1)*i))
            v=str(i+1)
            painter.drawText(space+50+5+10, 256+s//2-dy,s,s,Qt.AlignVCenter, v)

    painter.end()
    return qimage



