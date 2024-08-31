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

from tkinter.messagebox import NO
from PyQt5.QtWidgets import     QGraphicsView,QApplication,QVBoxLayout,\
                                QMenu,QAction, QWidget, QGraphicsItem,\
                                QScrollArea,QLabel,QMessageBox,QShortcut,QFileDialog
from PyQt5.QtCore import Qt,QPointF,QThread,pyqtSignal,QTimer
from PIL import Image,ImageDraw,ImageEnhance
import numpy as np
from PyQt5.QtGui import QPixmap,QImage,QCursor,QFont,QTransform,QKeySequence,QColor,QPainter
import os,copy
from mpicker_item import Cross,Circle,LongCross,give_colorbar,Line
from mpicker_core import show_3d, get_area, get_stretch, coord_global2local, show_3d_texture
from scipy.ndimage import median_filter
from multiprocessing import Process
from mpicker_show3dhelp import UI_Mpicker_Show3dhelp
from Mpicker_check import Mpicker_check
from Mpicker_particles import ParticleData
from mpicker_classeditor import Mpicker_classeditor
from mpicker_epickergui import Mpicker_Epickergui
import warnings
import mrcfile
for_fake_import = False
if for_fake_import:
    import Mpicker_gui


class Mpicker_ResultView(QGraphicsView):
    def __init__(self, *__args):
        super(Mpicker_ResultView, self).__init__(*__args)
        self.transform      = QTransform()
        self.Button_font    = QFont("Agency FB", 9)
        self.draw_flag      = False
        self.area_flag      = False
        self.stretch_flag   = False
        #Set short cut
        self.Shortcut_AddPoint = QShortcut(QKeySequence('S'), self)
        self.Shortcut_AddPointDown = QShortcut(QKeySequence('W'), self)
        self.Shortcut_NextPoint = QShortcut(QKeySequence('X'), self)
        self.Shortcut_DeletePoint = QShortcut(QKeySequence('Ctrl+D'), self)
        self.Shortcut_InvertPoint = QShortcut(QKeySequence('I'), self)
        self.Shortcut_AddPoint2 = QShortcut(QKeySequence('Ctrl+A'), self)
        self.Shortcut_DeletePoint2 = QShortcut(QKeySequence('Ctrl+Z'), self)
        self.Shortcut_PgUp = QShortcut(QKeySequence('PgUp'), self, context=Qt.WidgetShortcut)
        self.Shortcut_PgDown = QShortcut(QKeySequence('PgDown'), self, context=Qt.WidgetShortcut)
        self.Shortcut_Up = QShortcut(QKeySequence('Up'), self, context=Qt.WidgetShortcut)
        self.Shortcut_Down = QShortcut(QKeySequence('Down'), self, context=Qt.WidgetShortcut)
        self.Shortcut_Left = QShortcut(QKeySequence('Left'), self, context=Qt.WidgetShortcut)
        self.Shortcut_Right = QShortcut(QKeySequence('Right'), self, context=Qt.WidgetShortcut)
        self.Shortcut_Hide = QShortcut(QKeySequence('V'), self, context=Qt.WidgetShortcut)
        # Set ShortCut
        self.Shortcut_AddPoint.activated.connect(self.saveresultReferenceUp)
        self.Shortcut_AddPointDown.activated.connect(self.saveresultReferenceDown)
        self.Shortcut_NextPoint.activated.connect(self.nextresultPoint)
        self.Shortcut_DeletePoint.activated.connect(self.deleteresultReference)
        self.Shortcut_InvertPoint.activated.connect(self.resultReferenceInvert)
        self.Shortcut_AddPoint2.activated.connect(self.saveresultReference_2)
        self.Shortcut_DeletePoint2.activated.connect(self.deleteresultReference_2)
        self.Shortcut_PgUp.activated.connect(self.move_PgUp)
        self.Shortcut_PgDown.activated.connect(self.move_PgDown)
        self.Shortcut_Up.activated.connect(self.move_Up)
        self.Shortcut_Down.activated.connect(self.move_Down)
        self.Shortcut_Left.activated.connect(self.move_Left)
        self.Shortcut_Right.activated.connect(self.move_Right)
        self.Shortcut_Hide.activated.connect(self.HideresultPoint)

    def setParameters(self, UI):
        self.UI : Mpicker_gui.UI = UI
        self.UI.showResultImage = self.showResultImage
        self.UI.removeResultCursor = self.removeResultCursor
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.GraphicViewResultMenu)

        self.UI.Button_DeleteresultPoint.clicked.connect(self.deleteresultReference)
        # self.UI.Button_NextresultPoint.clicked.connect(self.nextresultPoint)
        self.UI.Button_saveresult.clicked.connect(self.SaveresultPoint)
        #self.UI.Button_loadresult.clicked.connect(self.LoadresultPoint)
        self.UI.Button_loadresult.clicked.connect(self.LoadresultReference)
        self.UI.Button_loadresultraw.clicked.connect(self.LoadresultReferenceRaw)
        self.UI.Button_showarea.clicked.connect(self.ShowresultArea)
        self.UI.Button_showstretch.clicked.connect(self.ShowresultStretch)
        self.UI.spinBox_radius.valueChanged.connect(self.SpinBox_radius)
        self.UI.spinBox_Zproject.valueChanged.connect(self.SpinBox_Zproject)
        self.UI.checkBox_showcircle2d.toggled.connect(lambda x: self.refresh_Points())
        # self.UI.horizontalSlider_radius.valueChanged.connect(self.slide_radius)
        self.UI.Button_hide.clicked.connect(self.HideresultPoint)
        self.UI.Button_show3d.clicked.connect(self.Show3D)
        self.UI.pushButton_OpenEditor.clicked.connect(self.OpenEditor)
        self.UI.pushButton_OpenEpicker.clicked.connect(self.OpenEpicker)
        self.UI.Button_showxyz.clicked.connect(self.ShowXYZ)
        self.HideFlag = False
        #self.UI.Button_Erase.clicked.connect(self.EraseresultPoint)
        self.UI.Button_Clear.clicked.connect(self.ClearresultPoint)
        self.UI.pushButton_flipz.clicked.connect(self.FlipZ)
        # self.EraseClickFlag = False
        # self.EraseFlag = False
        # Set Cursor pixmap
        # self.EraseCursorlen = 30
        # earse_path = os.path.abspath(os.path.dirname(__file__))
        # earse_file = os.path.join(earse_path,'Pic','EraseCursor.png')
        # pixmap = QPixmap(earse_file)
        # scaled_pixmap = pixmap.scaled(self.EraseCursorlen, self.EraseCursorlen)
        # self.cursorpix = QCursor(scaled_pixmap, -10, -10)
        #Max Gap length
        self.length = 10
        # Eraser
        self.EraseFlag = False
        self.PressFlag = False
        self.EraserCursor = self.create_eraser_cursor()
        self.timer_eraser = QTimer()
        self.timer_eraser.timeout.connect(self.eraser_func)
        self.UI.Button_eraser.clicked.connect(self.EraserClicked)

    def showResultImage(self):
        if not self.UI.allow_showResultImage:
            return
        self.area_flag = False # change to normal image
        self.stretch_flag = False
        self.UI.showResultImage_side(self.HideFlag)
        Zproject = int(self.UI.spinBox_Zproject.value())
        Zproject = min(self.UI.resultz - 1, self.UI.tomo_result.shape[0] - self.UI.resultz, Zproject)
        if Zproject >= 1:
            z1 = self.UI.resultz - 1 - Zproject
            z2 = self.UI.resultz + Zproject
            mrc_image_Image = self.UI.tomo_result[z1:z2,:,:].mean(axis=0)
        else:
            mrc_image_Image = self.UI.tomo_result[self.UI.resultz - 1, :, :]
        #mrc_image_Image = self.UI.result_tomo_min + self.UI.result_contrast * (mrc_image_Image - self.UI.old_result_tomo_min)
        mrc_image_Image = np.select(
            [(mrc_image_Image > self.UI.result_tomo_min) & (mrc_image_Image < self.UI.result_tomo_max),
             mrc_image_Image <= self.UI.result_tomo_min,
             mrc_image_Image >= self.UI.result_tomo_max],
            [self.UI.result_contrast * (mrc_image_Image - self.UI.result_tomo_min),
             0,
             255])
        mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
        # Show the Image
        mrc_image_Image = np.asarray(mrc_image_Image)
        height, width, channels = mrc_image_Image.shape
        bytesPerLine = channels * width
        qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.UI.resultpixmap = QPixmap.fromImage(qImg)
        self.UI.graphicsScene_result.clear()
        self.UI.graphicsScene_result.addPixmap(self.UI.resultpixmap)
        self.UI.screenshot_Fxy = self.UI.resultpixmap
        self.setScene(self.UI.graphicsScene_result)
        self.refresh_Points()
        self.refresh_Cursor()
        # show the coord
        if self.UI.CoordMapping is not None and self.UI.pixmap is not None:
            # if self.UI.label_Press_Flag == False:
            roundx = int(round(self.UI.resultx - 1))
            roundy = int(round(self.UI.resulty - 1))
            roundz = int(round(self.UI.resultz - 1))
            realz, realy, realx = self.UI.CoordMapping[:, roundz, roundy, roundx] + 1
            realz, realy, realx = int(round(realz)), int(round(realy)), int(round(realx))
            if realz < 1 or realz > self.UI.tomo_show.shape[0] \
                or realy < 1 or realy > self.UI.tomo_show.shape[1] \
                or realx < 1 or realx > self.UI.tomo_show.shape[2]:
                self.UI.saveoutofbound = True
            else:
                self.UI.realz, self.UI.realy, self.UI.realx = realz, realy, realx
                self.UI.saveoutofbound = False
            self.UI.showMrcImage(fromResult=True) # not remove cursor

                # 2020/7/20
                # if roundx < self.UI.CoordMapping.shape[3] and roundx >= 0 and\
                #         roundy < self.UI.CoordMapping.shape[2] and roundy >= 0 and\
                #         roundz < self.UI.CoordMapping.shape[1] and roundz >= 0:
                #     self.UI.realz, self.UI.realy, self.UI.realx = self.UI.CoordMapping[:, roundz, roundy, roundx]
                #     self.UI.realz = int(self.UI.realz + 1)
                #     self.UI.realx = round(self.UI.realx + 1)
                #     self.UI.realy = round(self.UI.realy + 1)
                #     if self.UI.realz >= 1 and self.UI.realz <= self.UI.tomo_show.shape[0] and\
                #         self.UI.realy >= 1 and self.UI.realy <= self.UI.tomo_show.shape[1] and\
                #         self.UI.realx >= 1 and self.UI.realx <= self.UI.tomo_show.shape[2]:
                #         self.UI.showMrcImage()
                #         self.UI.saveoutofbound = False
                #     else:
                #         self.UI.saveoutofbound = True

                    # else:
                    #     QMessageBox.warning(self, "Out of Bound",
                    #                     "Coord in result view has no relative coord in mrc view.",
                    #                     QMessageBox.Ok)
        # self.UI.label_Press_Flag = True
        self.UI.allow_showMrcImage = False # avoid use showMrcImage repeatly
        self.UI.allow_showResultImage = False
        self.UI.allow_showResultImage_side = False
        self.UI.doubleSpinBox_resultX.setValue(self.UI.resultx)
        self.UI.doubleSpinBox_resultY.setValue(self.UI.resulty)
        self.UI.doubleSpinBox_resultZ.setValue(self.UI.resultz)
        self.UI.spinBox_resultz.setValue(self.UI.resultz)
        self.UI.resulthorizontalSlider_z.setValue(self.UI.resultz)
        self.UI.allow_showMrcImage = True
        self.UI.allow_showResultImage = True
        self.UI.allow_showResultImage_side = True

    def removeResultCursor(self):
        self.UI.graphicsScene_result.clear()
        self.UI.graphicsScene_result.addPixmap(self.UI.resultpixmap)
        self.UI.screenshot_Fxy = self.UI.resultpixmap
        self.setScene(self.UI.graphicsScene_result)
        self.refresh_Points()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() & Qt.LeftButton:
            self.PressFlag = True
        if self.EraseFlag: # do not need to refresh position when eraser on
            return
        if self.UI.resultpixmap is not None:
            self.pressX = int(round(self.mapToScene(event.pos()).x()+1))
            self.pressY = int(self.UI.tomo_result.shape[1] - round(self.mapToScene(event.pos()).y()))
            # self.UI.label_Press_Flag = False
            if event.button() & Qt.LeftButton:
                if self.pressX > 0 and self.pressX <= self.UI.tomo_result.shape[2] and \
                        self.pressY > 0 and self.pressY <= self.UI.tomo_result.shape[1]:
                    # coord y is reversed in 3Dmod!
                    self.UI.resultx = self.pressX
                    self.UI.resulty = self.pressY
                    ## self.UI.label_Press_Flag = False
                    # self.EraseClickFlag = False
                    self.showResultImage()
            # elif event.button() & Qt.RightButton:
            #     if self.pressX > 0 and self.pressX <= self.UI.tomo_result.shape[2] and \
            #             self.pressY > 0 and self.pressY <= self.UI.tomo_result.shape[1]:
            #         self.EraseClickFlag = True
            ##         self.UI.label_Press_Flag = False
            # else:
            #     self.EraseClickFlag = False
                    
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() & Qt.LeftButton:
            self.PressFlag = False

    # def mouseReleaseEvent(self, event):
    #     super().mouseReleaseEvent(event)
    #     if self.UI.resultpixmap is not None and self.EraseFlag:
    #         if event.button() & Qt.RightButton:
    #             self.EraseClickFlag = False

    # def mouseMoveEvent(self, event):
    #     super().mouseMoveEvent(event)
    #     if self.UI.resultpixmap is not None and self.EraseFlag and self.EraseClickFlag:
    #         self.pressX = round(self.mapToScene(event.pos()).x() + 1)
    #         self.pressY = self.UI.tomo_result.shape[1] - round(self.mapToScene(event.pos()).y()) + 1
    #         select_item = self.UI.graphicsScene_result.itemAt(self.mapToScene(event.pos()).x(),
    #                                                           self.mapToScene(event.pos()).y(), self.transform)
    #         if select_item is not None:
    #             if select_item.type() == 65536:
    #                 self.deleteresultReference()

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        if self.EraseFlag: # do not need to refresh position when eraser on
            return
        if self.UI.resultpixmap is not None:
            self.pressX = int(round(self.mapToScene(event.pos()).x()+1))
            self.pressY = int(self.UI.tomo_result.shape[1] - round(self.mapToScene(event.pos()).y()))
            if event.button() & Qt.LeftButton:
                # Gap = self.length
                Gap = np.clip(self.UI.spinBox_radius.value(), 0, self.length)
                for i in range(len(self.UI.tomo_result_select)):
                    particle = self.UI.tomo_result_select[i]
                    if self.UI.resultz == particle.z and abs(self.pressX - particle.x) <= Gap \
                        and abs(self.pressY - particle.y) <= Gap:
                        # self.UI.label_Press_Flag = False
                        self.UI.tomo_result_current = i
                        self.ReText_select_result_scroll()
                        self.UI.resultx = particle.x
                        self.UI.resulty = particle.y
                        self.UI.resultz = particle.z   
                        self.showResultImage()
                        break

    def enterEvent(self, event):
        # if self.EraseFlag == False:
        #     QApplication.setOverrideCursor(Qt.CrossCursor)
        # else:
        #     QApplication.setOverrideCursor(self.cursorpix)
        super().enterEvent(event)
        if self.EraseFlag:
            QApplication.setOverrideCursor(self.EraserCursor)
        else:
            QApplication.setOverrideCursor(Qt.CrossCursor)

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
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)  # AnchorUnderMouse
        self.scale(1.1, 1.1)
        self.UI.graphicsView_resultside.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_resultside.scale(1.1, 1.1)
        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # self.scale(1.1,1.1)

    def zoomOut(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)  # AnchorUnderMouse
        self.scale(1 / 1.1, 1 / 1.1)
        self.UI.graphicsView_resultside.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.UI.graphicsView_resultside.scale(1 / 1.1, 1 / 1.1)
        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # self.scale(1/1.1,1/1.1)

    def move_PgUp(self):
        spinbox = self.UI.doubleSpinBox_resultZ
        spinbox.setValue(spinbox.value() + 1)

    def move_PgDown(self):
        spinbox = self.UI.doubleSpinBox_resultZ
        spinbox.setValue(spinbox.value() - 1)

    def move_Up(self):
        spinbox = self.UI.doubleSpinBox_resultY
        spinbox.setValue(spinbox.value() + 1)

    def move_Down(self):
        spinbox = self.UI.doubleSpinBox_resultY
        spinbox.setValue(spinbox.value() - 1)

    def move_Right(self):
        spinbox = self.UI.doubleSpinBox_resultX
        spinbox.setValue(spinbox.value() + 1)

    def move_Left(self):
        spinbox = self.UI.doubleSpinBox_resultX
        spinbox.setValue(spinbox.value() - 1)

    def refresh_Cursor(self):
        if self.UI.resultpixmap is not None:
            cross = Cross()
            cross.setPos(self.UI.resultx - 1, self.UI.tomo_result.shape[1] - self.UI.resulty)
            # if self.UI.label_Press_Flag == False and self.HideFlag == False:
            if self.HideFlag == False:
                cross.setColor("red")
                self.draw_flag = True
            else:
                cross.setColor("gray")
                self.draw_flag = False
            self.UI.graphicsScene_result.addItem(cross)

    def refresh_Points(self):
        if self.UI.resultpixmap is not None:
            self.select_group = self.UI.graphicsScene_result.createItemGroup([])
            # other classes (show)
            for idx in self.UI.particle_class_show:
                if idx == self.UI.particle_class_edit:
                    continue
                if idx not in self.UI.tomo_result_select_all.keys():
                    continue
                color_show = QColor(self.UI.class_file.idx_color(idx))
                for particle in self.UI.tomo_result_select_all[idx]:
                    drawx = particle.x - 1
                    drawy = self.UI.tomo_result.shape[1] - particle.y
                    if particle.z == round(self.UI.resultz):
                        if particle.UpDown() == "Up":
                            cross = Cross()
                        else:
                            cross = Cross(up=False)
                        cross.setPos(drawx ,drawy)
                        cross.setColor(color_show)
                        circle = Circle()
                        circle.setR(self.UI.draw_pix_radius)
                        circle.setPos(drawx,drawy)
                        circle.setColor(color_show)
                        self.select_group.addToGroup(cross)
                        self.select_group.addToGroup(circle)
                    elif self.UI.checkBox_showcircle2d.isChecked():
                        continue
                    elif abs(particle.z - self.UI.resultz) < self.UI.draw_pix_radius:
                        circle = Circle()
                        # radius = np.sqrt(self.UI.draw_pix_radius**2 - (particle.z - self.UI.resultz)**2)
                        circle.setRZ(self.UI.draw_pix_radius, abs(particle.z - self.UI.resultz))
                        circle.setPos(drawx, drawy)
                        circle.setColor(color_show)
                        self.select_group.addToGroup(circle)
            # the selected class (edit)
            color_edit = QColor(self.UI.class_file.idx_color(self.UI.particle_class_edit))
            for i, particle in enumerate(self.UI.tomo_result_select):
                drawx = particle.x - 1
                drawy = self.UI.tomo_result.shape[1] - particle.y
                if i == self.UI.tomo_result_current:
                    different_color = True
                else:
                    different_color = False

                if particle.z == round(self.UI.resultz):
                    if particle.UpDown() == "Up":
                        cross = Cross()
                    else:
                        cross = Cross(up=False)
                    cross.setPos(drawx ,drawy)
                    if different_color:
                        cross.setColor("different")
                    else:
                        cross.setColor(color_edit)

                    circle = Circle()
                    circle.setR(self.UI.draw_pix_radius)
                    circle.setPos(drawx,drawy)
                    if different_color:
                        circle.setColor("different")
                    else:
                        circle.setColor(color_edit)
                    self.select_group.addToGroup(cross)
                    self.select_group.addToGroup(circle)
                    if particle.has_point2():
                        drawx2 = particle.x2 - 1
                        drawy2 = self.UI.tomo_result.shape[1] - particle.y2
                        line = Line(drawx, drawy, drawx2, drawy2)
                        if different_color:
                            line.setColor("different")
                        else:
                            line.setColor(color_edit)
                        self.select_group.addToGroup(line)
                elif self.UI.checkBox_showcircle2d.isChecked():
                    continue
                elif abs(particle.z - self.UI.resultz) < self.UI.draw_pix_radius:
                    circle = Circle()
                    circle.setPos(drawx, drawy)
                    # radius = np.sqrt(self.UI.draw_pix_radius**2 - (particle.z - self.UI.resultz)**2)
                    circle.setRZ(self.UI.draw_pix_radius, abs(particle.z - self.UI.resultz))
                    if different_color:
                        circle.setColor("different")
                    else:
                        circle.setColor(color_edit)
                    self.select_group.addToGroup(circle)

            if self.HideFlag == True:
                self.select_group.hide()
            else:
                self.select_group.show()
            self.UI.graphicsScene_result.update()


    def GraphicViewResultMenu(self):
        if self.UI.resultpixmap is not None:
            # initial the Menu
            self.groupBox_menu = QMenu(self)
            self.actionA = QAction(u'Save as Reference(Up)', self)
            self.actionA.setShortcut("S")
            self.actionA.triggered.connect(self.saveresultReferenceUp)
            self.groupBox_menu.addAction(self.actionA)
            self.actionD = QAction(u'Save as Reference(Down)', self)
            self.actionD.setShortcut("W")
            self.actionD.triggered.connect(self.saveresultReferenceDown)
            self.groupBox_menu.addAction(self.actionD)         
            Gap = np.clip(self.UI.spinBox_radius.value(), 0, self.length)
            if len(self.UI.tomo_result_select) > 1:
                self.actionC = QAction(u'To Selected/Next Point', self)
                self.groupBox_menu.addAction(self.actionC)
                self.actionC.setShortcut("X")
                self.actionC.triggered.connect(self.nextresultPoint)
                self.actionH = QAction(u'Invert All Reference UpDown', self)
                self.groupBox_menu.addAction(self.actionH)
                self.actionH.triggered.connect(self.resultReferenceInvert_all)
                self.actionJ = QAction(u'Invert All Reference UpDown in this slice', self)
                self.groupBox_menu.addAction(self.actionJ)
                self.actionJ.triggered.connect(self.resultReferenceInvert_slice)
                self.actionI = QAction(u'Clear All Reference in this slice', self)
                self.groupBox_menu.addAction(self.actionI)
                self.actionI.triggered.connect(self.clearReference_slice)
                self.actionL = QAction(u'Clear All Point2 in this slice', self)
                self.groupBox_menu.addAction(self.actionL)
                self.actionL.triggered.connect(self.clearReference_2_slice)
                self.actionO = QAction(u'Convert all points into raw tomogram', self)
                self.groupBox_menu.addAction(self.actionO)
                self.actionO.triggered.connect(self.convertsultReference)       
            delete_point2 = False
            for particle in self.UI.tomo_result_select:
                if self.UI.resultz == particle.z and abs(self.pressX - particle.x) <= Gap \
                    and abs(self.pressY - particle.y) <= Gap:
                    self.actionB = QAction(u'Delete Reference', self)
                    self.actionB.setShortcut("Ctrl+D")
                    self.groupBox_menu.addAction(self.actionB)
                    self.actionB.triggered.connect(self.deleteresultReference_right)
                    self.actionE = QAction(u'Invert UpDown', self)
                    self.actionE.setShortcut("I")
                    self.actionE.triggered.connect(self.resultReferenceInvert_right)
                    self.groupBox_menu.addAction(self.actionE)
                    self.actionG = QAction(u'Delete Point2', self) # not the best way
                    self.actionG.setShortcut("Ctrl+Z")
                    self.actionG.triggered.connect(self.deleteresultReference_2_right)
                    delete_point2 = True
                    break
            if len(self.UI.tomo_result_select) > 0:
                particle_c = self.UI.tomo_result_select[self.UI.tomo_result_current]
                if self.UI.resultz == particle_c.z \
                    and (self.UI.resultx, self.UI.resulty) != (particle_c.x, particle_c.y):
                    self.actionF = QAction(u'Add Point2', self)
                    self.actionF.setShortcut("Ctrl+A")
                    self.actionF.triggered.connect(self.saveresultReference_2)
                    self.groupBox_menu.addAction(self.actionF)
                    self.actionK = QAction(u'Add Point2 for all in this slice', self)
                    self.groupBox_menu.addAction(self.actionK)
                    self.actionK.triggered.connect(self.saveresultReference_2_slice)
            if delete_point2:
                self.groupBox_menu.addAction(self.actionG)
            self.actionM = QAction(u'Save screenshot', self)
            self.actionM.triggered.connect(lambda : self.UI.save_pixmap(self.UI.screenshot_Fxy, "Flatten_xy"))
            self.groupBox_menu.addAction(self.actionM)         
            self.actionN = QAction(u'Add points uniformly in this slice', self)
            self.groupBox_menu.addAction(self.actionN)
            self.actionN.triggered.connect(self.saveresultReference_uniform)

            self.groupBox_menu.popup(QCursor.pos())

    def deleteresultReference_right(self):
        Gap = np.clip(self.UI.spinBox_radius.value(), 0, self.length)
        for i in range(len(self.UI.tomo_result_select)):
            particle = self.UI.tomo_result_select[i]
            if self.UI.resultz == particle.z and abs(self.pressX - particle.x) <= Gap \
                and abs(self.pressY - particle.y) <= Gap:
                del self.UI.tomo_result_select[i]
                #del self.UI.tomo_result_back[i]
                #if len(self.UI.tomo_result_select) >= 0:
                if i <= self.UI.tomo_result_current:
                    if self.UI.tomo_result_current != 0:
                        self.UI.tomo_result_current = self.UI.tomo_result_current - 1
                            # self.UI.resultx = self.UI.tomo_result_select[self.UI.tomo_result_current][0]
                            # self.UI.resulty = self.UI.tomo_result_select[self.UI.tomo_result_current][1]
                            # self.UI.resultz = self.UI.tomo_result_select[self.UI.tomo_result_current][2]
                        # else:
                        #     self.UI.tomo_result_current = 0
                            # self.UI.resultx = int(self.UI.tomo_result.shape[2] / 2)
                            # self.UI.resulty = self.UI.tomo_result.shape[1] - int(self.UI.tomo_result.shape[1] / 2)
                            # self.UI.resultz = int(self.UI.tomo_result.shape[0] / 2)
                # self.UI.label_Press_Flag = False
                self.Del_select_result_scroll(i)
                self.showResultImage()
                break
        
    # def preDeleteresultPoint(self):
    #         self.draw_flag = True
    #         self.deleteresultReference()

    def deleteresultReference(self):
        # depend on selected, not clicked. also used for button
        if self.UI.resultpixmap is not None and self.draw_flag:
            del_id = self.UI.tomo_result_current
            if len(self.UI.tomo_result_select) == 0:
                return
            elif len(self.UI.tomo_result_select) > 1:
                del self.UI.tomo_result_select[self.UI.tomo_result_current]
                #del self.UI.tomo_result_back[self.UI.tomo_result_current]
                if self.UI.tomo_result_current != 0:
                    self.UI.tomo_result_current = self.UI.tomo_result_current - 1
                else:
                    self.UI.tomo_result_current = len(self.UI.tomo_result_select) - 1
                # self.UI.resultx = self.UI.tomo_result_select[self.UI.tomo_result_current].x
                # self.UI.resulty = self.UI.tomo_result_select[self.UI.tomo_result_current].y
                # self.UI.resultz = self.UI.tomo_result_select[self.UI.tomo_result_current].z
            else:
                self.UI.tomo_result_select = []
                self.UI.tomo_result_current = 0
                # self.UI.resultx = int(self.UI.tomo_result.shape[2] / 2)
                # self.UI.resulty = self.UI.tomo_result.shape[1] - int(self.UI.tomo_result.shape[1] / 2)
                # self.UI.resultz = int(self.UI.tomo_result.shape[0] / 2)
            # self.UI.label_Press_Flag = False
            self.Del_select_result_scroll(del_id)
            self.showResultImage()

    def deleteresultReference_2_right(self):
        Gap = np.clip(self.UI.spinBox_radius.value(), 0, self.length)
        for i in range(len(self.UI.tomo_result_select)):
            particle = self.UI.tomo_result_select[i]
            if self.UI.resultz == particle.z and abs(self.pressX - particle.x) <= Gap \
                and abs(self.pressY - particle.y) <= Gap \
                and particle.has_point2():
                particle.del_point2()
                # self.UI.label_Press_Flag = False
                self.ReText_select_result_scroll()
                self.showResultImage()
                break   

    def deleteresultReference_2(self):
        if len(self.UI.tomo_result_select) > 0:
            self.UI.tomo_result_select[self.UI.tomo_result_current].del_point2()
            # self.UI.label_Press_Flag = False
            self.ReText_select_result_scroll()
            self.showResultImage()

    def resultReferenceInvert_right(self):
        Gap = np.clip(self.UI.spinBox_radius.value(), 0, self.length)
        for i in range(len(self.UI.tomo_result_select)):
            particle = self.UI.tomo_result_select[i]
            if self.UI.resultz == particle.z and abs(self.pressX - particle.x) <= Gap \
                and abs(self.pressY - particle.y) <= Gap:
                self.UI.tomo_result_select[i].invert_norm()
                self.UI.tomo_result_current = i
                # self.UI.label_Press_Flag = False
                self.ReText_select_result_scroll()
                self.showResultImage()
                break 

    def resultReferenceInvert(self):
        if len(self.UI.tomo_result_select) > 0:
            self.UI.tomo_result_select[self.UI.tomo_result_current].invert_norm()
            # self.UI.label_Press_Flag = False
            self.ReText_select_result_scroll()
            self.showResultImage()

    def resultReferenceInvert_all(self):
        if len(self.UI.tomo_result_select) > 0:
            for particle in self.UI.tomo_result_select:
                particle.invert_norm()
            # self.UI.label_Press_Flag = False
            self.ReText_select_result_scroll()
            self.showResultImage()

    def clearReference_slice(self):
        if len(self.UI.tomo_result_select) > 0:
            z_now = int(round(self.UI.resultz))
            reply = QMessageBox.question(self,
                'Clear Particles with z=' + str(z_now),
                'Do you want to clear all particles in this slice?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)
            if reply == QMessageBox.No:
                return
            particle_current = self.UI.tomo_result_select[self.UI.tomo_result_current]
            for i in range(len(self.UI.tomo_result_select) - 1, -1, -1):
                particle = self.UI.tomo_result_select[i]
                if z_now == particle.z:
                    self.UI.tomo_result_select.pop(i)
            self.UI.tomo_result_current = 0
            for i in range(len(self.UI.tomo_result_select)):
                if self.UI.tomo_result_select[i] == particle_current:
                    self.UI.tomo_result_current = i
                    break
            self.Reset_select_result_scroll()
            self.showResultImage()

    def clearReference_2_slice(self):
        if len(self.UI.tomo_result_select) > 0:
            z_now = int(round(self.UI.resultz))
            reply = QMessageBox.question(self,
                'Clear Point2 with z=' + str(z_now),
                'Do you want to clear all Point2 in this slice?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)
            if reply == QMessageBox.No:
                return
            for particle in self.UI.tomo_result_select:
                if z_now == particle.z:
                    particle.del_point2()
            self.ReText_select_result_scroll()
            self.showResultImage()

    def convertsultReference(self):
        if len(self.UI.tomo_result_select) > 0:
            if self.UI.tabWidget.currentIndex() != 1:
                self.UI.tabBarInit(1)
            # self.UI.tomo_manual_select = []
            # just add, not clear
            for particle in self.UI.tomo_result_select:
                x, y, z = particle.x, particle.y, particle.z
                if self.point_outof_bound(particle.x, particle.y, particle.z):
                    continue
                rz, ry, rx = self.UI.CoordMapping[:, z-1, y-1, x-1] + 1
                rx, ry, rz = int(round(rx)), int(round(ry)), int(round(rz))
                if [rx, ry, rz] not in self.UI.tomo_manual_select:
                        self.UI.tomo_manual_select.append([rx, ry, rz])
            # edit from LoadReference() in mpicker_framemanual.py
            self.UI.frame_manual.Reset_scrollArea_Select()
            self.UI.frame_manual.scrollArea_manualpoints_widget.setLayout(self.UI.frame_manual.scrollArea_manualpoints_vbox)
            self.UI.frame_manual.scrollArea_manualpoints.setWidget(self.UI.frame_manual.scrollArea_manualpoints_widget)
            self.UI.frame_manual.showMrcImage()

    def resultReferenceInvert_slice(self):
        if len(self.UI.tomo_result_select) > 0:
            z_now = int(round(self.UI.resultz))
            for particle in self.UI.tomo_result_select:
                if z_now == particle.z:
                    particle.invert_norm()
            self.ReText_select_result_scroll()
            self.showResultImage()
            
    def ClearresultPoint(self):
        if len(self.UI.tomo_result_select) == 0:
            return
        reply = QMessageBox.question(self,
                'Clear Particles',
                'Do you want to clear all particles?\n(you still need press "Save All" to really save it)',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.UI.tomo_result_select = []
            self.UI.tomo_result_current = 0
            # self.UI.label_Press_Flag = False
            self.Reset_select_result_scroll()
            self.showResultImage()

    def FlipZ(self):
        if self.UI.resultpixmap is None:
            return
        reply = QMessageBox.question(self,
                'FlipZ',
                'Do you want to FlipZ (RotY 180)?\n The picked particles will be saved to the file.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
        if reply == QMessageBox.No:
            return
        
        self.UI.tomo_result = None
        self.UI.tomo_addresult = None # it should have been useless when flipz. mmap may conflict with open?
        path_tomo = self.UI.showResult_path
        with mrcfile.open(path_tomo, permissive=True) as mrc:
            tomo = mrc.data[::-1, :, ::-1].copy()
            voxel_size = mrc.voxel_size
        with mrcfile.new(path_tomo, overwrite=True) as mrc:
            mrc.set_data(tomo)
            mrc.voxel_size = voxel_size
        tomo = None

        self.UI.CoordMapping = None
        path_fnpy = self.UI.showResult_path.replace("_result.mrc", "_convert_coord.npy")
        fnpy = np.load(path_fnpy)[:, ::-1, :, ::-1]
        np.save(path_fnpy, fnpy)
        fnpy = None

        # copy from scroll_check in Mpicker_gui
        with mrcfile.mmap(path_tomo) as mrc:
            self.UI.tomo_result = mrc.data[:, ::-1, :]
        self.UI.CoordMapping = np.load(path_fnpy)

        self.SaveresultPoint(flipz=True) # update coords
        self.Reset_select_result_scroll()
        self.showResultImage()

    def saveresultReference(self, up=True):
        if self.UI.resultpixmap is not None and self.draw_flag:
            if not self.UI.saveoutofbound:
                # same_flag = 0
                x, y, z = round(self.UI.resultx), round(self.UI.resulty), round(self.UI.resultz)
                particle=ParticleData(x, y, z, self.UI.particle_class_edit)
                if not up:
                    particle.invert_norm() # default particle.up=1
                # for select in self.UI.tomo_result_select:
                #     if select[0] == self.UI.resultx and select[1] == self.UI.resulty and select[2] == self.UI.resultz:
                #         same_flag = 1
                #         break
                if particle not in self.UI.tomo_result_select:
                    #if self.UI.pixmap is not None:
                        #self.UI.tomo_result_back.append([self.UI.realx,self.UI.realy,self.UI.realz])
                    #self.UI.Draw_the_result_point()
                    #self.UI.tomo_result_select.append([self.UI.resultx, self.UI.resulty, self.UI.resultz])
                    self.UI.tomo_result_select.append(particle)
                    self.UI.tomo_result_current = len(self.UI.tomo_result_select) - 1
                    self.Append_select_result_scroll()
                    #self.UI.select_result_scroll_widget.setLayout(self.UI.select_result_scroll_vbox)
                    #self.UI.select_result_scroll.setWidget(self.UI.select_result_scroll_widget)
                    # self.UI.label_Press_Flag = False
                    self.showResultImage()
                    #self.refresh_Points()
            else:
                QMessageBox.information(self, 'Save Points Warning',
                                        'The Point you save is not of bound in original raw file.\n'
                                        'Please select points within the original raw file', QMessageBox.Ok)
    
    def saveresultReferenceUp(self):
        return self.saveresultReference(up=True)

    def saveresultReferenceDown(self):
        return self.saveresultReference(up=False)

    def saveresultReference_2(self):
        if self.UI.resultpixmap is not None and self.draw_flag \
            and len(self.UI.tomo_result_select) > 0:
            if self.UI.saveoutofbound:
                QMessageBox.information(self, 'Save Points Warning',
                                        'The Point you save is not of bound in original raw file.\n'
                                        'Please select points within the original raw file', QMessageBox.Ok)
                return
            particle_c = self.UI.tomo_result_select[self.UI.tomo_result_current]
            if self.UI.resultz == particle_c.z \
                and (self.UI.resultx, self.UI.resulty) != (particle_c.x, particle_c.y):
                particle_c.del_point2()
                particle_c.add_point2(self.UI.resultx, self.UI.resulty, self.UI.resultz)
                self.ReText_select_result_scroll()
                # self.UI.label_Press_Flag = False
                self.showResultImage()

    def point_outof_bound(self, x, y, z):
        if self.UI.resultpixmap is None:
            return True
        x, y, z = int(x), int(y), int(z)
        if x<1 or x>self.UI.CoordMapping.shape[3] or y<1 or y>self.UI.CoordMapping.shape[2] \
            or z<1 or z>self.UI.CoordMapping.shape[1]:
            return True
        realz, realy, realx = self.UI.CoordMapping[:, z-1, y-1, x-1] + 1
        if realx < 1 or realx > self.UI.tomo_show.shape[2] or realy < 1 or realy > self.UI.tomo_show.shape[1] \
            or realz < 1 or realz > self.UI.tomo_show.shape[0]:
            return True
        return False

    def saveresultReference_2_slice(self):
        if self.UI.resultpixmap is not None and self.draw_flag \
            and len(self.UI.tomo_result_select) > 0:
            if self.UI.saveoutofbound:
                QMessageBox.information(self, 'Save Points Warning',
                                        'The Point you save is not of bound in original raw file.\n'
                                        'Please select points within the original raw file', QMessageBox.Ok)
                return
            particle_c = self.UI.tomo_result_select[self.UI.tomo_result_current]
            if self.UI.resultz == particle_c.z \
                and (self.UI.resultx, self.UI.resulty) != (particle_c.x, particle_c.y):
                particle_c.add_point2(self.UI.resultx, self.UI.resulty, self.UI.resultz)
                dx, dy = particle_c.x2 - particle_c.x, particle_c.y2 - particle_c.y
                z_now = particle_c.z
                for particle in self.UI.tomo_result_select:
                    if z_now != particle.z or particle.has_point2():
                        continue
                    x2, y2 = particle.x + dx, particle.y + dy
                    if self.point_outof_bound(x2, y2, z_now):
                        continue
                    particle.add_point2(x2, y2, z_now)
                self.ReText_select_result_scroll()
                self.showResultImage()

    def saveresultReference_uniform(self):
        if self.UI.resultpixmap is not None and self.draw_flag:
            distance = int(self.UI.spinBox_radius.value())
            reply = QMessageBox.question(self,
                    'Add Particles',
                    f'Do you want to add particles in the interval of {distance} pixel?\n'+
                    '(you can change the interval by "Radius size")',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No)
            if reply == QMessageBox.Yes:
                edit_idx = self.UI.particle_class_edit
                z = int(round(self.UI.resultz))
                sx, sy = self.UI.tomo_result.shape[2], self.UI.tomo_result.shape[1]
                for x in range(1, sx+1, distance):
                    for y in range(1, sy+1, distance):
                        if self.point_outof_bound(x, y, z):
                            continue
                        particle = ParticleData(x, y, z, edit_idx)
                        if particle not in self.UI.tomo_result_select:
                            self.UI.tomo_result_select.append(particle)
                self.Reset_select_result_scroll()
                self.showResultImage()

    def LoadresultReference(self):
        if self.UI.resultpixmap is None:
            return
        if self.UI.pickfolder is not None and os.path.isdir(self.UI.pickfolder):
            from_dir = self.UI.pickfolder
        else:
            from_dir = os.getcwd()
        fname = QFileDialog.getOpenFileName(self, "Open Coord File (xyz, start from 1)", from_dir)[0]
        if fname == '':
            return
        #tomo_result_back_old = copy.deepcopy(self.UI.tomo_result_back)
        self.load_coord_file(fname)

    def load_coord_file(self, fname):
        tomo_result_select_old = copy.deepcopy(self.UI.tomo_result_select)
        edit_idx = self.UI.particle_class_edit
        try:
            coords = np.loadtxt(fname, ndmin=2)
            if coords.shape[1] == 2:
                # only xy
                xy_xyz_full = 0
                z = int(round(self.UI.resultz))
            elif coords.shape[1] < 16:
                # only xyz
                xy_xyz_full = 1
            else:
                # full data list
                xy_xyz_full = 2
            for data in coords:
                if xy_xyz_full == 0:
                    x, y = data
                    particle = ParticleData(x, y, z, edit_idx)
                elif xy_xyz_full == 1:
                    x, y, z = data[:3]
                    particle = ParticleData(x, y, z, edit_idx)
                else:
                    particle = ParticleData(data)
                    particle.set_class(edit_idx)
                    particle.clear_calculate()
                if particle.z <= self.UI.tomo_result.shape[0] and particle.z >= 1 \
                    and particle.y <= self.UI.tomo_result.shape[1] and particle.y >= 1 \
                    and particle.x <= self.UI.tomo_result.shape[2] and particle.x >= 1:
                    #z_raw, y_raw, x_raw = self.UI.CoordMapping[:, z-1, y-1, x-1] + 1
                    #z_raw, y_raw, x_raw = int(round(z_raw)), round(y_raw), round(x_raw)
                    if particle not in self.UI.tomo_result_select:
                        #self.UI.tomo_result_back.append([x_raw, y_raw, z_raw])
                        self.UI.tomo_result_select.append(particle)
        except Exception as e:
            #self.UI.tomo_result_back = tomo_result_back_old
            self.UI.tomo_result_select = tomo_result_select_old
            print(e)
            QMessageBox.warning(self, "Input Warning", "your input file is not right", QMessageBox.Ok)

        # copy from saveresultReference(self)
        self.Reset_select_result_scroll()
        #self.UI.select_result_scroll_widget.setLayout(self.UI.select_result_scroll_vbox)
        #self.UI.select_result_scroll.setWidget(self.UI.select_result_scroll_widget)
        # self.UI.label_Press_Flag = False
        self.showResultImage()

    def LoadresultReferenceRaw(self):
        # order of input coords will change
        if self.UI.resultpixmap is None:
            return
        if self.UI.pickfolder is not None and os.path.isdir(self.UI.pickfolder):
            from_dir = self.UI.pickfolder
        else:
            from_dir = os.getcwd()
        fname = QFileDialog.getOpenFileName(self, "Open Coord File (xyz, start from 1)", from_dir)[0]
        if fname == '':
            return
        #tomo_result_back_old = copy.deepcopy(self.UI.tomo_result_back)
        tomo_result_select_old = copy.deepcopy(self.UI.tomo_result_select)
        edit_idx = self.UI.particle_class_edit
        try:
            coords = np.loadtxt(fname, ndmin=2)
            zyxglobal = [] # start from 0
            for x, y, z in coords:
                if z <= self.UI.tomo_show.shape[0] and z >= 1 and y <= self.UI.tomo_show.shape[1] and y >= 1 \
                    and x <= self.UI.tomo_show.shape[2] and x >= 1:
                    if [z-1, y-1, x-1] not in zyxglobal:
                        zyxglobal.append([z-1, y-1, x-1])
            zyxglobal = np.array(zyxglobal)
            interval = np.linalg.norm(self.UI.CoordMapping[:, 1, 0, 0] - self.UI.CoordMapping[:, 0, 0, 0])
            zyxlocal = coord_global2local(zyxglobal, self.UI.CoordMapping, tomoshape=self.UI.tomo_show.shape, max_dist=interval*2)
            if zyxlocal.ndim == 1:
                zyxlocal = np.expand_dims(zyxlocal, axis=0)
            for z,y,x in zyxlocal:
                # so, xyz in global is float float int, in local is int int int ? by yxf 
                x, y, z = round(x) + 1, round(y) + 1, round(z) + 1
                #z_raw, y_raw, x_raw = self.UI.CoordMapping[:, z-1, y-1, x-1] + 1
                #z_raw, y_raw, x_raw = int(round(z_raw)), round(y_raw), round(x_raw)
                particle=ParticleData(x,y,z,edit_idx)
                if particle not in self.UI.tomo_result_select:
                    #self.UI.tomo_result_back.append([x_raw, y_raw, z_raw])
                    self.UI.tomo_result_select.append(particle)
        except Exception as e:
            #self.UI.tomo_result_back = tomo_result_back_old
            self.UI.tomo_result_select = tomo_result_select_old
            print(e)
            QMessageBox.warning(self, "Input Warning", "your input file is not right", QMessageBox.Ok)

        # copy from saveresultReference(self)
        self.Reset_select_result_scroll()
        #self.UI.select_result_scroll_widget.setLayout(self.UI.select_result_scroll_vbox)
        #self.UI.select_result_scroll.setWidget(self.UI.select_result_scroll_widget)
        # self.UI.label_Press_Flag = False
        self.showResultImage()

    def mixcolorimage(self, weight_list, color_list):
        # weight_list is the list of 2d array(same shape as image), range from 0 to 1. 
        # color_list is the list of color, such as [255,0,0]
        # return QImage (RGB888)
        num=len(weight_list)
        result=np.zeros((weight_list[0].shape[0], weight_list[0].shape[1], 3))
        for i in range(num):
            result += np.expand_dims(weight_list[i], axis=2) * np.array(color_list[i])
        result=result.astype(np.uint8)
        height, width, channels = result.shape
        bytesPerLine = channels * width
        return QImage(result.data, width, height, bytesPerLine, QImage.Format_RGB888)

    def deleteMask(self, mgrid):
        z, y, x = mgrid + 1
        mask = (z<1) | (z>self.UI.tomo_show.shape[0]) | \
                (y<1) | (y>self.UI.tomo_show.shape[1]) | \
                (x<1) | (x>self.UI.tomo_show.shape[2])
        return mask

    def addColorbar(self, mode, max_v):
        # mode = 'area' or 'stretch'
        sceneRect=self.UI.graphicsScene_result.sceneRect()
        colorbar_img=give_colorbar(mode, maxv=max_v)
        colorbar=self.UI.graphicsScene_result.addPixmap(QPixmap.fromImage(colorbar_img))
        colorbar_x=self.UI.tomo_result.shape[2]+10
        # colorbar_y=self.UI.tomo_result.shape[1]-colorbar_img.height()
        colorbar.setPos(colorbar_x ,0)
        try:
            colorbar.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        except:
            print("cannot set QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations")
        self.UI.graphicsScene_result.setSceneRect(sceneRect)

    def ShowresultArea(self):
        if self.UI.resultpixmap is None:
            return
        if self.area_flag is True: # already show area
            self.showResultImage() # change back to normal image
        else:
            self.area_flag = True
            self.stretch_flag = False
            # self.UI.label_Press_Flag = False # just make the cross red
            area_min, area_max = 1, 3 # to control contrast ##change
            filter_size = 0
            interval = np.linalg.norm(self.UI.CoordMapping[:, 1, 0, 0] - self.UI.CoordMapping[:, 0, 0, 0])
            def rescaleImg(a,amin,amax):
                return np.select([(a>amin)&(a<amax), a<=amin, a>=amax],[255/(amax-amin)*(a-amin), 0, 255])

            # xy slice. mrc_image_Image's xyz is different from real mrc
            mgrid_slice = self.UI.CoordMapping[:, int(round(self.UI.resultz) - 1), ::-1, :] 
            mrc_image_Image = get_area(mgrid_slice) / interval**2 # normalize to area=1 as standard
            if filter_size > 0:
                mrc_image_Image= median_filter(mrc_image_Image, size=filter_size)
            mrc_image_Image[mrc_image_Image==0] = 1e-3
            image_expand = np.zeros_like(mrc_image_Image)
            image_expand[mrc_image_Image<1] = 1/mrc_image_Image[mrc_image_Image<1] - 1
            image_expand = rescaleImg(image_expand,area_min-1,area_max-1)/255
            image_expand[self.deleteMask(mgrid_slice)] = 0
            image_contract = np.zeros_like(mrc_image_Image)
            image_contract[mrc_image_Image>=1] = mrc_image_Image[mrc_image_Image>=1] - 1
            image_contract = rescaleImg(image_contract,area_min-1,area_max-1)/255
            image_contract[self.deleteMask(mgrid_slice)] = 0
            image_real = self.UI.tomo_result[self.UI.resultz - 1, :, :]
            image_real = rescaleImg(image_real,self.UI.result_tomo_min,self.UI.result_tomo_max)/255
            #image=np.array([image_real*0.5+image_contract*0.5,image_real*0.5,image_real*0.5+image_expand*0.5])*255
            # image=np.array([image_real*0.5+(1-image_expand)*0.5,image_real*0.5+(1-image_contract-image_expand)*0.5,image_real*0.5+(1-image_contract)*0.5])*255
            # image=image.transpose(1,2,0).copy().astype(np.uint8)
            # height, width, channels = image.shape
            # bytesPerLine = channels * width
            # qimage_rgb = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            weight_list = [image_real*(1-image_expand-image_contract), image_expand, image_contract]
            color_list = [(255,255,255),(0,0,255),(255,0,0)]
            qimage_rgb=self.mixcolorimage(weight_list, color_list)
            # copy from showResultImage
            self.UI.resultpixmap = QPixmap.fromImage(qimage_rgb)
            self.UI.graphicsScene_result.clear()
            self.UI.graphicsScene_result.addPixmap(self.UI.resultpixmap)
            self.UI.screenshot_Fxy = self.UI.resultpixmap
            self.addColorbar('area',area_max)
            self.setScene(self.UI.graphicsScene_result)
            self.refresh_Points()
            self.refresh_Cursor()

            # xz slice. mrc_image_Image's xyz is different from real mrc
            mgrid_slice = self.UI.CoordMapping[:, :, ::-1, int(round(self.UI.resultx-1))].transpose((0, 2, 1))
            mrc_image_Image = get_area(mgrid_slice) / interval**2
            mrc_image_Image[mrc_image_Image==0] = 1e-3
            image_expand = np.zeros_like(mrc_image_Image)
            image_expand[mrc_image_Image<1] = 1/mrc_image_Image[mrc_image_Image<1] - 1
            image_expand = rescaleImg(image_expand,area_min-1,area_max-1)/255
            image_expand[self.deleteMask(mgrid_slice)] = 0
            image_contract = np.zeros_like(mrc_image_Image)
            image_contract[mrc_image_Image>=1] = mrc_image_Image[mrc_image_Image>=1] - 1
            image_contract = rescaleImg(image_contract,area_min-1,area_max-1)/255
            image_contract[self.deleteMask(mgrid_slice)] = 0
            image_real = self.UI.tomo_result[:, :, int(round(self.UI.resultx-1))].transpose((1, 0))
            # round(x-1) != round(x)-1  ...hhh
            image_real = rescaleImg(image_real,self.UI.result_tomo_min,self.UI.result_tomo_max)/255
            weight_list = [image_real*(1-image_expand-image_contract), image_expand, image_contract]
            color_list = [(255,255,255),(0,0,255),(255,0,0)]
            qimage_rgb=self.mixcolorimage(weight_list, color_list)
            # copy from showResultImage_side
            self.UI.sidepixmap = QPixmap.fromImage(qimage_rgb)
            self.UI.graphicsScene_resultside.clear()
            self.UI.graphicsScene_resultside.addPixmap(self.UI.sidepixmap)
            self.UI.screenshot_Fyz = self.UI.sidepixmap
            self.UI.graphicsView_resultside.setScene(self.UI.graphicsScene_resultside)
            longcross = LongCross()
            longcross.setPos(self.UI.resultz - 1, self.UI.tomo_result.shape[1] - self.UI.resulty)
            # if self.UI.label_Press_Flag == False:
            #     longcross.setColor("red")
            # else:
            #     longcross.setColor("gray")
            self.UI.graphicsScene_resultside.addItem(longcross)

    def ShowresultStretch(self):
        if self.UI.resultpixmap is None:
            return
        if self.stretch_flag is True: # already show stretch
            self.showResultImage() # change back to normal image
        else:
            self.stretch_flag = True
            self.area_flag    = False
            # self.UI.label_Press_Flag = False # just make the cross red
            area_min, area_max = 1, 3 # to control contrast. long axis / short axis, not area here ##change
            filter_size = 0
            def rescaleImg(a,amin,amax):
                return np.select([(a>amin)&(a<amax), a<=amin, a>=amax],[255/(amax-amin)*(a-amin), 0, 255])
                
            # xy slice. mrc_image_Image's xyz is different from real mrc
            mgrid_slice = self.UI.CoordMapping[:, int(round(self.UI.resultz) - 1), ::-1, :] 
            mrc_image_Image = get_stretch(mgrid_slice)
            if filter_size > 0:
                mrc_image_Image= median_filter(mrc_image_Image, size=filter_size)
            image_stretch = mrc_image_Image - 1
            image_stretch = rescaleImg(image_stretch,area_min-1,area_max-1)/255
            image_stretch[self.deleteMask(mgrid_slice)] = 0
            image_real = self.UI.tomo_result[self.UI.resultz - 1, :, :]
            image_real = rescaleImg(image_real,self.UI.result_tomo_min,self.UI.result_tomo_max)/255
            weight_list = [image_real*(1-image_stretch), image_stretch]
            color_list = [(255,255,255), (255,0,0)]
            qimage_rgb=self.mixcolorimage(weight_list, color_list)
            # copy from showResultImage
            self.UI.resultpixmap = QPixmap.fromImage(qimage_rgb)
            self.UI.graphicsScene_result.clear()
            self.UI.graphicsScene_result.addPixmap(self.UI.resultpixmap)
            self.UI.screenshot_Fxy = self.UI.resultpixmap
            self.addColorbar('stretch',area_max)
            self.setScene(self.UI.graphicsScene_result)
            self.refresh_Points()
            self.refresh_Cursor()

            # xz slice. mrc_image_Image's xyz is different from real mrc
            mgrid_slice = self.UI.CoordMapping[:, :, ::-1, int(round(self.UI.resultx-1))].transpose((0, 2, 1))
            mrc_image_Image = get_stretch(mgrid_slice)
            image_stretch = mrc_image_Image - 1
            image_stretch = rescaleImg(image_stretch,area_min-1,area_max-1)/255
            image_stretch[self.deleteMask(mgrid_slice)] = 0
            image_real = self.UI.tomo_result[:, :, int(round(self.UI.resultx-1))].transpose((1, 0))
            image_real = rescaleImg(image_real,self.UI.result_tomo_min,self.UI.result_tomo_max)/255
            weight_list = [image_real*(1-image_stretch), image_stretch]
            color_list = [(255,255,255), (255,0,0)]
            qimage_rgb=self.mixcolorimage(weight_list, color_list)
            # copy from showResultImage_side
            self.UI.sidepixmap = QPixmap.fromImage(qimage_rgb)
            self.UI.graphicsScene_resultside.clear()
            self.UI.graphicsScene_resultside.addPixmap(self.UI.sidepixmap)
            self.UI.screenshot_Fyz = self.UI.sidepixmap
            self.UI.graphicsView_resultside.setScene(self.UI.graphicsScene_resultside)
            longcross = LongCross()
            longcross.setPos(self.UI.resultz - 1, self.UI.tomo_result.shape[1] - self.UI.resulty)
            # if self.UI.label_Press_Flag == False:
            #     longcross.setColor("red")
            # else:
            #     longcross.setColor("gray")
            self.UI.graphicsScene_resultside.addItem(longcross)

    def nextresultPoint(self):
        if len(self.UI.tomo_result_select) >= 1:
            Gap = np.clip(self.UI.spinBox_radius.value(), 0, self.length)
            particle = self.UI.tomo_result_select[self.UI.tomo_result_current]
            if self.UI.resultz == particle.z and abs(self.UI.resultx - particle.x) <= Gap \
                and abs(self.UI.resulty - particle.y) <= Gap :
                # move to next point
                if self.UI.tomo_result_current == len(self.UI.tomo_result_select) - 1:
                    self.UI.tomo_result_current = 0
                else:
                    self.UI.tomo_result_current += 1
                self.ReText_select_result_scroll()
            else:
                # move to this point
                pass
            self.UI.resultx = self.UI.tomo_result_select[self.UI.tomo_result_current].x
            self.UI.resulty = self.UI.tomo_result_select[self.UI.tomo_result_current].y
            self.UI.resultz = self.UI.tomo_result_select[self.UI.tomo_result_current].z   
            # self.pressX = self.UI.resultx
            # self.pressY = self.UI.resulty
            self.showResultImage()
            #self.refresh_Points()

    def Reset_select_result_scroll(self):
        if self.UI.resultpixmap is not None:
            for del_i in reversed(range(self.UI.select_result_scroll_vbox.count())):
                self.UI.select_result_scroll_vbox.itemAt(del_i).widget().setParent(None)
            for select_i in range(len(self.UI.tomo_result_select)):
                select_label = QLabel()
                # select_label.setText(
                #     f"{select_i + 1}. [{self.UI.tomo_result_back[select_i][0]},"
                #                     f"{self.UI.tomo_result_back[select_i][1]},"
                #                     f"{self.UI.tomo_result_back[select_i][2]}]")
                if self.UI.tomo_result_select[select_i].has_point2():
                    string = self.UI.tomo_result_select[select_i].UpDown(LR=True) + "2"
                else:
                    string = self.UI.tomo_result_select[select_i].UpDown(LR=True) + "1"
                select_label.setText( "%d: %d, %d, %d, %s"%(
                    select_i + 1,
                    self.UI.tomo_result_select[select_i].x,
                    self.UI.tomo_result_select[select_i].y,
                    self.UI.tomo_result_select[select_i].z,
                    string
                ) )
                Font = QFont("Agency FB", 9)
                if select_i == self.UI.tomo_result_current:
                    Font.setBold(True)
                select_label.setFont(Font)
                self.UI.select_result_scroll_vbox.addWidget(select_label)

    def ReText_select_result_scroll(self):
        if self.UI.resultpixmap is not None:
            for select_i, particle in enumerate(self.UI.tomo_result_select):
                Font = QFont("Agency FB", 9)
                if select_i == self.UI.tomo_result_current:
                    Font.setBold(True)
                if particle.has_point2():
                    string = particle.UpDown(LR=True) + "2"
                else:
                    string = particle.UpDown(LR=True) + "1"
                text = "%d: %d, %d, %d, %s"%(
                    select_i + 1, particle.x, particle.y, particle.z, string)
                self.UI.select_result_scroll_vbox.itemAt(select_i).widget().setText(text)
                self.UI.select_result_scroll_vbox.itemAt(select_i).widget().setFont(Font)

    def Append_select_result_scroll(self):
        if self.UI.resultpixmap is not None:
            select_label = QLabel()
            self.UI.select_result_scroll_vbox.addWidget(select_label)
            self.ReText_select_result_scroll()

    def Del_select_result_scroll(self, del_id):
        if self.UI.resultpixmap is not None:
            item = self.UI.select_result_scroll_vbox.takeAt(del_id)
            item.widget().setParent(None) # necessary
            self.ReText_select_result_scroll()

    # def NextresultPoint(self):
    #     if self.UI.resultpixmap is not None and self.draw_flag:
    #         if len(self.UI.tomo_result_select) >= 1:
    #             self.UI.tomo_result_current = self.UI.tomo_result_current + 1
    #             if self.UI.tomo_result_current >= len(self.UI.tomo_result_select):
    #                 self.UI.tomo_result_current = 0
    #             self.Reset_select_result_scroll()
    #             self.UI.resultx = self.UI.tomo_result_select[self.UI.tomo_result_current][0]
    #             self.UI.resulty = self.UI.tomo_result_select[self.UI.tomo_result_current][1]
    #             self.UI.resultz = self.UI.tomo_result_select[self.UI.tomo_result_current][2]
    #             self.showResultImage()
    #             #self.refresh_Points()

    def SaveresultPoint(self, flipz=False):
        # save to _SelectPoints.txt
        # flipz means you already flipped the flattened tomo and npy, so you need to save the flipped coords.
        if self.UI.resultpixmap is not None:
            self.UI.tomo_result_select_all[self.UI.particle_class_edit] = copy.deepcopy(self.UI.tomo_result_select)
            if len(self.UI.tomo_result_select) == 0:
                self.UI.tomo_result_select_all.pop(self.UI.particle_class_edit)
                if len(self.UI.tomo_result_select_all) == 0:
                    self.UI.tomo_result_select_all = {1:[]}

            if flipz:
                sx = self.UI.tomo_result.shape[2]
                sz = self.UI.tomo_result.shape[0]
                for particle in self.UI.tomo_result_select:
                    particle.flipz(sx, sz)
                for class_idx in self.UI.tomo_result_select_all.keys():
                    for particle in self.UI.tomo_result_select_all[class_idx]:
                        particle.flipz(sx, sz)

            result_array = []
            for class_idx in sorted(self.UI.tomo_result_select_all.keys()):
                tomo_result_select_1 = self.UI.tomo_result_select_all[class_idx]
                if len(tomo_result_select_1) > 0:
                    result_array += [particle.final_list(self.UI.CoordMapping) for particle in tomo_result_select_1]
            result_array = np.array(result_array)
            self.Reset_class_label()
            if len(result_array) > 0:
                np.savetxt(self.UI.savetxt_path, result_array,
                            fmt=ParticleData.fmt(), header=ParticleData.header())
            else:
                clearfile = open(self.UI.savetxt_path, "w").close()
                # with open(self.UI.savetxt_path, "w") as f:
                #     for item in self.UI.tomo_result_select:
                #         # print("item = ",item)
                #         f.write(f"{item[0]}, {item[1]}, {item[2]}")
                #         xcoord = round(item[0]-1)
                #         ycoord = round(item[1]-1)
                #         zcoord = round(item[2]-1)
                #         ori_coord = self.UI.CoordMapping[:,zcoord,ycoord,xcoord]
                #         x_ori  = round(ori_coord[2]+1,2)
                #         y_ori  = round(ori_coord[1]+1,2)
                #         z_ori  = round(ori_coord[0]+1)
                #         f.write(f"\t{x_ori}, {y_ori}, {z_ori}\n")

    def LoadresultPoint(self):
        # load from _SelectPoints.txt
        # print("Load result Point")
        if self.UI.tomo_result is not None:
            if os.path.isfile(self.UI.savetxt_path) == False:
                makefile = open(self.UI.savetxt_path,"w").close()
            try:
                warnings.filterwarnings("ignore")
                # ignore empty file warning
                datas = np.loadtxt(self.UI.savetxt_path, ndmin=2)
                warnings.filterwarnings("default")
                for data in datas:
                    particle = ParticleData(data)
                    if particle.z > self.UI.tomo_result.shape[0] or particle.z < 1 \
                        or particle.y > self.UI.tomo_result.shape[1] or particle.y < 1 \
                        or particle.x > self.UI.tomo_result.shape[2] or particle.x < 1:
                        continue
                    particle.clear_calculate()
                    class_idx = particle.class_idx
                    if class_idx not in self.UI.tomo_result_select_all.keys():
                        self.UI.tomo_result_select_all[class_idx] = []
                    if particle not in self.UI.tomo_result_select_all[class_idx]:
                        self.UI.tomo_result_select_all[class_idx].append(particle)
            except:
                print("read old fasion coords")
                with open(self.UI.savetxt_path, "r") as f:
                    datas = f.readlines()
                for line in datas:
                    twocoord = line.split("\t")
                    Procoords = twocoord[0].split(', ')
                    Procoord = [float(n) for n in Procoords]
                    particle = ParticleData(Procoord[0], Procoord[1], Procoord[2])
                    # particle.class_idx should be 1 by default
                    if particle not in self.UI.tomo_result_select_all[1]:
                        self.UI.tomo_result_select_all[1].append(particle)
                    # Procoord[2] = round(Procoord[2])
                    # Oricoords = twocoord[1].split(', ')
                    # Oricoord = [float(n) for n in Oricoords]
                    # Oricoord[2] = round(Oricoord[2])
                    # same_flag = 0
                    # for select in self.UI.tomo_result_select:
                    #     if select[0] == Procoord[0] and select[1] == Procoord[1] and select[2] == Procoord[2]:
                    #         same_flag = 1
                    #         break
                    # if same_flag == 0:
                    #     self.UI.tomo_result_select.append([Procoord[0],Procoord[1],Procoord[2]])
                    #     self.UI.tomo_result_back.append([Oricoord[0],Oricoord[1],Oricoord[2]])
            # self.UI.tomo_result_current = len(self.UI.tomo_result_select) - 1
            # if self.UI.tomo_result_current < 0:
            #     self.UI.tomo_result_current = 0
            if self.UI.particle_class_edit in self.UI.tomo_result_select_all.keys():
                self.UI.tomo_result_select = copy.deepcopy(self.UI.tomo_result_select_all[self.UI.particle_class_edit])
            else:
                self.UI.tomo_result_select = []
            self.UI.tomo_result_current = 0
            # self.UI.label_Press_Flag = False
            self.Reset_select_result_scroll()
            self.Reset_class_label()
            self.UI.select_result_scroll_widget.setLayout(self.UI.select_result_scroll_vbox)
            self.UI.select_result_scroll.setWidget(self.UI.select_result_scroll_widget)
            #self.showResultImage()
            self.refresh_Points()
            #self.UI.remove_result_point()

    # def EraseresultPoint(self):
    #     if self.EraseFlag == False:
    #         if self.HideFlag:
    #             QMessageBox.information(self, 'Reminder',
    #                                     'You still hide your selected points\n'
    #                                     'Please show them before you erase.',QMessageBox.Ok)
    #         self.EraseFlag = True
    #         self.customContextMenuRequested.disconnect()
    #         self.UI.Button_Erase.setStyleSheet("background-color: gray")
    #     else:
    #         self.EraseFlag = False
    #         self.customContextMenuRequested.connect(self.GraphicViewResultMenu)
    #         self.UI.Button_Erase.setStyleSheet("background-color: None")

    def HideresultPoint(self):
        if self.UI.resultpixmap is None:
            return
        if self.select_group.isVisible():
            self.HideFlag = True
            self.UI.Button_hide.setText("Show")
            self.UI.Button_hide.setFont(self.Button_font)
            self.UI.Button_hide.setStyleSheet("background-color: gray")
        else:
            self.HideFlag = False
            # self.UI.label_Press_Flag = False
            self.UI.Button_hide.setText("Hide")
            self.UI.Button_hide.setFont(self.Button_font)
            self.UI.Button_hide.setStyleSheet("background-color: None")
        self.showResultImage()

    def EraserClicked(self):
        if self.UI.resultpixmap is None:
            return
        if self.EraseFlag:
            self.EraseFlag = False
            self.UI.Button_eraser.setStyleSheet("background-color: None")
            self.eraser_enable_other(True)
            self.timer_eraser.stop()
        else:
            if not self.select_group.isVisible():
                self.HideresultPoint() # will showResultImage too
            else:
                self.showResultImage() # to cancel area and stretch
            self.EraseFlag = True
            self.UI.Button_eraser.setStyleSheet("background-color: gray")
            self.eraser_enable_other(False)
            self.timer_eraser.start(10) # 10ms

    def eraser_enable_other(self, enabled):
        # disable some function when eraser is on
        self.UI.Button_hide.setEnabled(enabled)
        self.UI.doubleSpinBox_resultX.setEnabled(enabled)
        self.UI.doubleSpinBox_resultY.setEnabled(enabled)
        self.Shortcut_Up.setEnabled(enabled)
        self.Shortcut_Down.setEnabled(enabled)
        self.Shortcut_Left.setEnabled(enabled)
        self.Shortcut_Right.setEnabled(enabled)
        self.Shortcut_Hide.setEnabled(enabled)
        self.UI.Button_Next.setEnabled(enabled)
        self.UI.Button_Remove.setEnabled(enabled)
        self.UI.Button_Check.setEnabled(enabled)
        self.UI.Button_Openraw.setEnabled(enabled)
        self.UI.Button_OpenSession.setEnabled(enabled)
        self.UI.Button_Selectsavepath.setEnabled(enabled)
        if enabled:
            self.setDragMode(self.ScrollHandDrag)
        else:
            self.setDragMode(self.NoDrag)

    def create_eraser_cursor(self):
        pixmap = QPixmap(10, 10)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(Qt.red))
        painter.drawEllipse(2, 2, 6, 6)
        painter.end()
        return QCursor(pixmap)

    def eraser_func(self):
        if self.UI.resultpixmap is None:
            return
        if not self.PressFlag:
            return
        pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        pressX = int(round(pos.x() + 1))
        pressY = int(self.UI.tomo_result.shape[1] - round(pos.y()))
        Gap2 = self.UI.spinBox_radius.value() ** 2
        changed = False
        for i in reversed(range(len(self.UI.tomo_result_select))):
            particle = self.UI.tomo_result_select[i]
            if self.UI.resultz != particle.z:
                continue
            if (pressX - particle.x)**2 + (pressY - particle.y)**2 <= Gap2:
                del self.UI.tomo_result_select[i]
                if i <= self.UI.tomo_result_current and self.UI.tomo_result_current > 0:
                    self.UI.tomo_result_current -= 1
                self.Del_select_result_scroll(i)
                changed = True
        if changed:
            self.UI.graphicsScene_result.removeItem(self.select_group)
            # removeItem should have been applied in refresh_Points
            self.refresh_Points()

    def SpinBox_radius(self):
        if self.UI.resultpixmap is not None:
            self.set_radius(self.UI.spinBox_radius.value())

    def SpinBox_Zproject(self):
        if self.UI.resultpixmap is not None:
            self.showResultImage()

    # def slide_radius(self,value):
    #     if self.UI.resultpixmap is not None:
    #         self.set_radius(value)

    def set_radius(self,value):
        self.UI.spinBox_radius.setValue(value)
        # self.UI.horizontalSlider_radius.setValue(value)
        self.UI.draw_pix_radius = value
        self.refresh_Points()

    def Show3D(self):
        if self.UI.resultpixmap is not None:
            #show 3d gui
            self.show3dhelp = UI_Mpicker_Show3dhelp()
            #self.memseg.setParameters(self)
            self.Thread_show3d = QThread_Show3d()
            self.Thread_show3d.setParameters(self.UI)
            self.Thread_show3d.error_signal.connect(self.errorprocess)
            self.Thread_show3d.start()

        else:
            QMessageBox.warning(self, "Input Warning",
                                "Nothing to show. "
                                "Please open a result before showing the 3d model.",
                                QMessageBox.Ok)

    def ShowXYZ(self):
        if self.UI.showResult_path is not None and self.UI.resultpixmap is not None:
            show_name           = os.path.basename(self.UI.showResult_path)
            self.UI.show_mrc    = self.UI.showResult_path
            self.UI.show_xyz_flag = True
            self.check          = Mpicker_check()
            #self.check.Init_show_mrc()
            self.check.setParameters(self.UI)
            self.check.setWindowTitle(show_name)
            self.check.show()

    def OpenEditor(self):
        if self.UI.showResult_path is not None and self.UI.resultpixmap is not None:
            if self.UI.class_file.no_file:
                return
            self.editor = Mpicker_classeditor(self.UI.class_file.file_path)
            self.editor.load_file()
            self.editor.apply_signal.connect(self.ClassChange)
            self.editor.show()

    def OpenEpicker(self):
        if self.UI.showResult_path is not None and self.UI.resultpixmap is not None:
            try:
                self.epicker.show()
                self.epicker.activateWindow()
            except:
                self.epicker = Mpicker_Epickergui()
                self.epicker.setParameters(self.UI)
                self.epicker.show()

    def ClassChange(self):
        if self.UI.showResult_path is None or self.UI.resultpixmap is None:
            # if you open new tomo but editor still open, should not happen
            return
        # save list to dict
        self.UI.tomo_result_select_all[self.UI.particle_class_edit] = copy.deepcopy(self.UI.tomo_result_select)
        # remove from dict if it is cleaned
        if len(self.UI.tomo_result_select) == 0:
            self.UI.tomo_result_select_all.pop(self.UI.particle_class_edit)
            # but dict cannot be empty 
            if len(self.UI.tomo_result_select_all) == 0:
                self.UI.tomo_result_select_all = {1:[]}
        # read file to update
        self.UI.update_class_file()
        # get list from dict, if it has
        if self.UI.particle_class_edit in self.UI.tomo_result_select_all.keys():
            self.UI.tomo_result_select = copy.deepcopy(self.UI.tomo_result_select_all[self.UI.particle_class_edit])
        else:
            self.UI.tomo_result_select = []
        # selected particle cannot out of range
        if self.UI.tomo_result_current > len(self.UI.tomo_result_select) - 1:
            self.UI.tomo_result_current = 0
        # refresh particles info to be shown
        self.Reset_select_result_scroll()
        # refresh selected class info to be shown
        self.Reset_class_label()
        # refresh image and repaint circle...
        # self.UI.label_Press_Flag = False
        self.showResultImage()

    def Reset_class_label(self):
        idx_now = self.UI.particle_class_edit
        idx_max = max(self.UI.tomo_result_select_all.keys())
        color = self.UI.class_file.idx_color(idx_now)
        name = self.UI.class_file.idx_name(idx_now)
        text = " %d / %d : %s"%(idx_now, idx_max, name)
        self.UI.label_ClassEdit.setText(text)
        self.UI.label_ClassEdit.setStyleSheet(f"color: {color}; background: silver")

    def errorprocess(self,bool):
        if bool:
            str_e = repr(self.Thread_show3d.exception)
            QMessageBox.warning(self, "Show 3d Error",
                                "Can not show the 3d model of result. "+ str_e,
                                QMessageBox.Ok)


class QThread_Show3d(QThread):
    def __init__(self):
        super(QThread_Show3d,self).__init__()
        self.exitcode = False
        self.exception = None

    error_signal    = pyqtSignal(bool)

    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        if self.UI.advance_setting.checkBox_pointsin3d.isChecked() and len(self.UI.tomo_result_select) > 0:
            for particle in self.UI.tomo_result_select:
                particle.calculate(self.UI.CoordMapping)
            points_zyx = [ [particle.Rz - 1, particle.Ry - 1, particle.Rx - 1] for 
                            particle in self.UI.tomo_result_select ] # start from 0
        else:
            points_zyx = None
        points_radius = self.UI.spinBox_radius.value()
        proc=Process(target=show_3d_texture,args=(self.UI.showResult_path,
                                          self.UI.showResult_path.replace("_result.mrc", "_convert_coord.npy"),
                                          self.UI.tomo_show.shape, points_zyx, points_radius,
                                          True, self.UI.advance_setting.spinBox_adshow3d_bin.value()))
        try:
            proc.start()
            proc.join()
        except Exception as e:
            self.exitcode = True
            self.exception = e
            self.error_signal.emit(True)

    def kill_thread(self):
        self.terminate()