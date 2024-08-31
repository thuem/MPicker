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

from PyQt5.QtWidgets import     QGraphicsView,QApplication,QCheckBox,\
                                QMenu,QAction,QMessageBox,\
                                QVBoxLayout,QWidget,QLabel,QFrame,\
                                QShortcut,QGraphicsScene,QHBoxLayout,\
                                QFileDialog
from PyQt5.QtGui import QPixmap,QImage,QCursor,QFont,QKeySequence
from PyQt5.QtCore import Qt,QRectF,QThread,pyqtSignal
from PIL import Image,ImageDraw,ImageEnhance
from scipy.ndimage import binary_dilation
import numpy as np
import mrcfile
import copy
import os
from mpicker_item import Cross,Circle
import warnings
import Mpicker_core_gui

class Mpicker_FrameManual(QFrame):
    def __init__(self, *__args):
        super(Mpicker_FrameManual, self).__init__(*__args)

    def setParameters(self, UI):
        self.UI = UI
        self.UI.Button_next_manual.clicked.connect(self.nextPoint)
        self.UI.Button_Savenew.clicked.connect(self.SaveManualNewPoints)
        self.UI.Button_Delete_manual.clicked.connect(self.DeleteReference)
        self.UI.Button_Clear_manual.clicked.connect(self.ClearReference)
        # self.UI.Button_Saveold.clicked.connect(self.SaveManualOldPoints)
        self.UI.Button_Loadcoord.clicked.connect(self.LoadReference)
        #self.customContextMenuRequested.connect(self.GraphicViewMenu)
        self.UI.spinBox_ManualStretchZ.valueChanged.connect(self.refresh_Points)
        self.scrollArea_manualpoints = self.UI.scrollArea_manualpoints
        self.scrollArea_manualpoints_vbox = QVBoxLayout()
        self.scrollArea_manualpoints_widget = QWidget()
        self.scrollArea_manualpoints_widget.mousePressEvent = self.scrollArea_manualpoints_mousePressEvent
        # Manual Points
        self.scrollArea_manualsurf = self.UI.scrollArea_manualsurf
        self.scrollArea_manualsurf_vbox = self.UI.scrollArea_manualsurf_vbox
        self.scrollArea_manualsurf_widget = QWidget()
        self.scrollArea_manualsurf_widget.mousePressEvent = self.scrollArea_manualsurf_mousePressEvent
        # Cross Cursor
        self.scrollArea_manualsurf_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.scrollArea_manualsurf_widget.customContextMenuRequested.connect(self.CheckBox_manual_Menu)
        # Gap and line width setting
        self.length = 5
        # self.width = 2
        self.number = None

    def showMrcImage(self, fromResult=False):
        if self.UI.tomo_show is not None and self.UI.allow_showMrcImage:
            if self.UI.realz <= self.UI.tomo_show.shape[0] and self.UI.realz >= 1:
                mrc_image_Image = self.UI.tomo_show[self.UI.realz - 1, :, :]
                # mrc_image_data  = self.UI.tomo_data[self.UI.realz - 1, :, :]
                # mrc_Image_sum   = np.sum(mrc_image_Image)
                # mrc_data_sum    = np.sum(mrc_image_data)
                # if self.UI.show_mrc_flag == True and mrc_Image_sum == mrc_data_sum:
                if self.UI.show_mrc_flag == True and self.UI.tomo_show is self.UI.tomo_data:
                    mrc_image_Image = np.select(
                        [(mrc_image_Image > self.UI.mrc_tomo_min) & (mrc_image_Image < self.UI.mrc_tomo_max),
                         mrc_image_Image <= self.UI.mrc_tomo_min,
                         mrc_image_Image >= self.UI.mrc_tomo_max],
                        [self.UI.mrc_contrast * (mrc_image_Image - self.UI.mrc_tomo_min),
                         0,
                         255])
                else: # tomo_show is mask or boundary, binary
                    tomo_min = np.min(mrc_image_Image).astype(float)
                    tomo_max = np.max(mrc_image_Image).astype(float)
                    if tomo_max != tomo_min:
                        mrc_image_Image = (mrc_image_Image - tomo_min) / (tomo_max - tomo_min) * 255
                    else:
                        mrc_image_Image = np.zeros_like(mrc_image_Image)
                mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
                # Draw the extract line
                mrc_image_Image = np.asarray(mrc_image_Image)
                height, width, channels = mrc_image_Image.shape
                bytesPerLine = channels * width
                qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.UI.pixmap = QPixmap.fromImage(qImg)
                self.UI.graphicsScene.clear()
                self.UI.graphicsScene.addPixmap(self.UI.pixmap)
                self.UI.graphicsView.setScene(self.UI.graphicsScene)

                self.refresh_Points()
                self.refresh_Cursor(fromResult)
                self.CheckSaveButton()

                # if self.UI.label_Press_Flag == True:
                #     if self.UI.CoordMapping is not None and self.UI.resultpixmap is not None:
                #         self.UI.allow_showMrcImage = False
                #         self.UI.showResultImage()
                if not fromResult and self.UI.resultpixmap is not None and self.UI.sidepixmap is not None:
                    # not so necessary in fact...
                    self.UI.removeResultCursor()
                    self.UI.removeResultCursor_side()

                self.UI.allow_showMrcImage = False
                self.UI.horizontalSlider_z.setValue(self.UI.realz)
                self.UI.doubleSpinBox_X.setValue(self.UI.realx)
                self.UI.doubleSpinBox_Y.setValue(self.UI.realy)
                self.UI.doubleSpinBox_Z.setValue(self.UI.realz)
                self.UI.spinBox_z.setValue(self.UI.realz)
                self.UI.doubleSpinBox_Contrast.setValue(self.UI.slider_Contrast_value)
                self.UI.doubleSpinBox_Bright.setValue(self.UI.slider_Bright_value)
                self.UI.allow_showMrcImage = True
                #self.UI.doubleSpinBox_LContrast


    def refresh_Points(self):
        stretchz = max(1, self.UI.spinBox_ManualStretchZ.value())
        if self.UI.pixmap is not None:
            self.select_group = self.UI.graphicsScene.createItemGroup([])
            if self.UI.tomo_manual_show is not None:
                for manual in self.UI.tomo_manual_show:
                    for pos in manual:
                        drawx = pos[0] - 1
                        drawy = self.UI.tomo_show.shape[1] - pos[1]
                        if pos[2] == self.UI.realz:
                            cross = Cross()
                            cross.setPos(drawx, drawy)
                            cross.setColor("blue")
                            circle = Circle()
                            circle.setR(self.UI.radius_surf)
                            circle.setPos(drawx, drawy)
                            circle.setColor("blue")
                            self.select_group.addToGroup(cross)
                            self.select_group.addToGroup(circle)
                        else:
                            if abs(pos[2]-self.UI.realz) < self.UI.radius_surf * stretchz:
                                circle = Circle()
                                circle.setPos(drawx, drawy)
                                circle.setColor("blue")
                                radius = self.UI.radius_surf - abs(pos[2] - self.UI.realz) / stretchz
                                circle.setR(radius)
                                self.select_group.addToGroup(circle)
            for pos in self.UI.tomo_manual_select:
                drawx = pos[0] - 1
                drawy = self.UI.tomo_show.shape[1] - pos[1]
                if pos[2] == self.UI.realz:
                    cross = Cross()
                    cross.setPos(drawx ,drawy)
                    cross.setColor("green")
                    circle = Circle()
                    circle.setR(self.UI.radius_surf)
                    circle.setPos(drawx,drawy)
                    circle.setColor("green")
                    self.select_group.addToGroup(cross)
                    self.select_group.addToGroup(circle)
                else:
                    if abs(pos[2]-self.UI.realz) < self.UI.radius_surf * stretchz:
                        circle = Circle()
                        circle.setPos(drawx, drawy)
                        circle.setColor("green")
                        radius = self.UI.radius_surf - abs(pos[2] - self.UI.realz) / stretchz
                        circle.setR(radius)
                        self.select_group.addToGroup(circle)
            self.UI.graphicsScene.update()

    def refresh_Cursor(self, fromResult=False):
        if fromResult and self.UI.saveoutofbound:
            return
        if self.UI.pixmap is not None:
            cross = Cross()
            cross.setPos(self.UI.realx - 1, self.UI.tomo_show.shape[1] - self.UI.realy)
            self.UI.graphicsScene.addItem(cross)

    def scrollArea_manualpoints_mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            x = event.x()
            y = event.y()
            for i in range(self.scrollArea_manualpoints_vbox.count()):
                chosed_label = self.scrollArea_manualpoints_vbox.itemAt(i).widget()
                pic_xmin, pic_ymin = chosed_label.x(), chosed_label.y()
                pic_xmax, pic_ymax = chosed_label.x() + chosed_label.width(), chosed_label.y() + chosed_label.height()
                if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                    try:
                        self.UI.tomo_manual_current = i
                        self.UI.realx = self.UI.tomo_manual_select[self.UI.tomo_manual_current][0]
                        self.UI.realy = self.UI.tomo_manual_select[self.UI.tomo_manual_current][1]
                        self.UI.realz = self.UI.tomo_manual_select[self.UI.tomo_manual_current][2]
                        break
                    except:
                        pass
            self.ReBold_scrollArea_Select()
            self.showMrcImage()

    def GraphicViewMenu(self):
        if self.UI.pixmap is not None and self.UI.tabWidget.currentWidget().objectName() == "tab_manual":
            self.groupBox_menu = QMenu(self)
            self.actionA = QAction(u'Save as Reference', self)
            self.actionA.setShortcut("Shift+S")
            self.groupBox_menu.addAction(self.actionA)
            self.actionA.triggered.connect(self.saveReference)
            Gap = self.length
            if len(self.UI.tomo_manual_select) > 1:
                self.actionC = QAction(u'Next Point', self)
                self.actionC.setShortcut("Shift+X")
                self.groupBox_menu.addAction(self.actionC)
                self.actionC.triggered.connect(self.nextPoint)
            for select in self.UI.tomo_manual_select:
                if self.UI.realz == select[2] and self.UI.pressX <= select[0] + Gap and self.UI.pressX >= select[
                    0] - Gap and self.UI.pressY <= select[1] + Gap and self.UI.pressY > select[1] - Gap:
                    self.actionB = QAction(u'Delete Reference', self)
                    self.actionB.setShortcut("Shift+D")
                    self.groupBox_menu.addAction(self.actionB)
                    self.actionB.triggered.connect(self.deleteReference)
            self.actionD = QAction(u'Save screenshot', self)
            self.actionD.triggered.connect(lambda : self.UI.save_pixmap(self.UI.pixmap, "Raw_xy"))
            self.groupBox_menu.addAction(self.actionD)
            
            self.groupBox_menu.popup(QCursor.pos())

    def saveReference(self):
        same_flag = 0
        for select in self.UI.tomo_manual_select:
            if select[0] == self.UI.realx and select[1] == self.UI.realy and select[2] == self.UI.realz:
                same_flag = 1
                break
        if same_flag == 0:
            self.UI.tomo_manual_select.append([self.UI.realx, self.UI.realy, self.UI.realz])
            self.UI.tomo_manual_current = len(self.UI.tomo_manual_select) - 1
            # Add a point
            self.Reset_scrollArea_Select()
            self.scrollArea_manualpoints_widget.setLayout(self.scrollArea_manualpoints_vbox)
            self.scrollArea_manualpoints.setWidget(self.scrollArea_manualpoints_widget)
            # self.UI.label_Press_Flag = False
            self.showMrcImage()
            # self.refresh_Points()

    def deleteReference(self):
        Gap = self.length
        for i in range(len(self.UI.tomo_manual_select)):
            select = self.UI.tomo_manual_select[i]
            if self.UI.realz == select[2] and self.UI.pressX <= select[0] + Gap and self.UI.pressX >= select[
                0] - Gap and self.UI.pressY <= select[1] + Gap and self.UI.pressY > select[1] - Gap:
                self.number = self.scrollArea_manualpoints_vbox.itemAt(i).widget().text().split(". ")[0]
                del self.UI.tomo_manual_select[i]
                if len(self.UI.tomo_manual_select) > 0:
                    if i <= self.UI.tomo_manual_current:
                        if self.UI.tomo_manual_current != 0:
                            self.UI.tomo_manual_current = self.UI.tomo_manual_current - 1
                        else:
                            self.UI.tomo_manual_current = 0
                    self.Reset_scrollArea_Select()
                else:
                    self.Reset_scrollArea_Select()
                break
        #self.remove_surf_config()
        #self.remove_extract_config()
        self.showMrcImage()

    def DeleteReference(self):
        if self.UI.pixmap is not None:# and self.draw_flag:
            if len(self.UI.tomo_manual_select) >= 1:
                del self.UI.tomo_manual_select[self.UI.tomo_manual_current]
                if self.UI.tomo_manual_current != 0:
                    self.UI.tomo_manual_current = self.UI.tomo_manual_current - 1
                    self.UI.realx = self.UI.tomo_manual_select[self.UI.tomo_manual_current][0]
                    self.UI.realy = self.UI.tomo_manual_select[self.UI.tomo_manual_current][1]
                    self.UI.realz = self.UI.tomo_manual_select[self.UI.tomo_manual_current][2]
                else:
                    if len(self.UI.tomo_manual_select) > 0:
                        self.UI.tomo_manual_current = len(self.UI.tomo_manual_select) - 1
                        self.UI.realx = self.UI.tomo_manual_select[self.UI.tomo_manual_current][0]
                        self.UI.realy = self.UI.tomo_manual_select[self.UI.tomo_manual_current][1]
                        self.UI.realz = self.UI.tomo_manual_select[self.UI.tomo_manual_current][2]
                    else:
                        self.UI.tomo_manual_current = 0
                        self.UI.realx = int(self.UI.tomo_show.shape[2] / 2)
                        self.UI.realy = self.UI.tomo_show.shape[1] - int(self.UI.tomo_show.shape[1] / 2)
                        self.UI.realz = int(self.UI.tomo_show.shape[0] / 2)
                self.Reset_scrollArea_Select()
                self.showMrcImage()
    
    def LoadReference(self):
        if self.UI.pixmap is None:
            return
        fname = QFileDialog.getOpenFileName(self, "Open Coord File (xyz, start from 1)", os.getcwd())[0]
        if fname == '':
            return
        tomo_manual_select_old = copy.deepcopy(self.UI.tomo_manual_select)
        try:
            coords = np.loadtxt(fname, ndmin=2)
            for x, y, z in coords:
                x, y, z = float(x), float(y), int(round(z))
                if z <= self.UI.tomo_show.shape[0] and z >= 1 and y <= self.UI.tomo_show.shape[1] and y >= 1 \
                    and x <= self.UI.tomo_show.shape[2] and x >= 1:
                    if [x, y, z] not in self.UI.tomo_manual_select:
                        self.UI.tomo_manual_select.append([x, y, z])
        except Exception as e:
            self.UI.tomo_manual_select = tomo_manual_select_old
            print(e)
            QMessageBox.warning(self, "Input Warning", "your input file is not right", QMessageBox.Ok)
        #print(self.UI.tomo_manual_select)

        # copy from saveReference(self)
        self.Reset_scrollArea_Select()
        self.scrollArea_manualpoints_widget.setLayout(self.scrollArea_manualpoints_vbox)
        self.scrollArea_manualpoints.setWidget(self.scrollArea_manualpoints_widget)
        # self.UI.label_Press_Flag = False
        self.showMrcImage()

    def nextPoint(self):
        if len(self.UI.tomo_manual_select) > 1:
            if self.UI.tomo_manual_current == len(self.UI.tomo_manual_select) - 1:
                self.UI.tomo_manual_current = 0
            else:
                self.UI.tomo_manual_current += 1
            # show the pos
            self.Reset_scrollArea_Select()
            self.UI.realx = self.UI.tomo_manual_select[self.UI.tomo_manual_current][0]
            self.UI.realy = self.UI.tomo_manual_select[self.UI.tomo_manual_current][1]
            self.UI.realz = self.UI.tomo_manual_select[self.UI.tomo_manual_current][2]
            self.showMrcImage()

    def ClearReference(self):
        if self.UI.pixmap is not None:
            self.UI.tomo_manual_current = 0
            self.UI.tomo_manual_select = []
            for del_i in reversed(range(self.scrollArea_manualpoints_vbox.count())):
                self.scrollArea_manualpoints_vbox.itemAt(del_i).widget().setParent(None)
            self.showMrcImage()

    def Reset_scrollArea_Select(self):
        #if self.UI.pixmap is not None:
            for del_i in reversed(range(self.scrollArea_manualpoints_vbox.count())):
                self.scrollArea_manualpoints_vbox.itemAt(del_i).widget().setParent(None)
            for select_i in range(len(self.UI.tomo_manual_select)):
                select_label = QLabel()
                select_label.setText(
                    f"{select_i + 1}. [{self.UI.tomo_manual_select[select_i][0]},"
                                    f"{self.UI.tomo_manual_select[select_i][1]},"
                                    f"{self.UI.tomo_manual_select[select_i][2]}]")
                Font = QFont("Agency FB", 9)
                if select_i == self.UI.tomo_manual_current:
                    Font.setBold(True)
                select_label.setFont(Font)
                self.scrollArea_manualpoints_vbox.addWidget(select_label)
            #print("manual = ",self.UI.tomo_manual_select)

    def ReBold_scrollArea_Select(self):
        for select_i in range(self.scrollArea_manualpoints_vbox.count()):
            Font = QFont("Agency FB", 9)
            if select_i == self.UI.tomo_manual_current:
                Font.setBold(True)
            self.scrollArea_manualpoints_vbox.itemAt(select_i).widget().setFont(Font)

    def CheckSaveButton(self):
        if len(self.UI.tomo_manual_select) > 0:
            self.UI.Button_Savenew.setEnabled(True)
        else:
            self.UI.Button_Savenew.setEnabled(False)
        # if len(self.UI.tomo_manual_surf) > 0:
        #     self.UI.Button_Saveold.setEnabled(True)
        # else:
        #     self.UI.Button_Saveold.setEnabled(False)

    def SaveManualNewPoints(self):
        if self.UI.pixmap is not None:
            if self.UI.Text_SavePath.toPlainText() != "":
                try:
                    self.UI.tomo_manual_surf.append(copy.deepcopy(self.UI.tomo_manual_select))
                    #self.UI.tomo_manual_surf.append(self.UI.tomo_manual_select)
                    #print("self.UI.tomo_manual_surf = \n",self.UI.tomo_manual_surf)
                    self.UI.tomo_manual_surfcurrent = len(self.UI.tomo_manual_surf) - 1
                    # Add a point
                    self.UI.frame_Setting.config_set_select_manual(False)
                    self.Reset_scrollArea_Surf_Select()
                    self.UI.tomo_manual_number = self.UI.tomo_manual_number + 1
                    #print("self.UI.tomo_manual_number = ",self.UI.tomo_manual_number)
                    self.saveConfigFile()
                    self.saveTxtFile()
                except Exception as e:
                    str_e = repr(e)
                    QMessageBox.warning(self, "Input Warning", str_e, QMessageBox.Ok)
            else:
                QMessageBox.warning(self,"Input Warning","Please Select a Save Path before generate manual surface."
                                                        ,QMessageBox.Ok)


    # def SaveManualOldPoints(self):
    #     if self.UI.pixmap is not None:
    #         if self.UI.Text_SavePath.toPlainText() != "":
    #             try:
    #                 self.UI.tomo_manual_surf[self.UI.tomo_manual_surfcurrent] = copy.deepcopy(self.UI.tomo_manual_select)
    #                 self.saveTxtFile()
    #                 self.saveConfigFile()
    #                 self.showMrcImage()
    #             except Exception as e:
    #                 str_e = repr(e)
    #                 QMessageBox.warning(self, "SaveManualOldPoints Warning", str_e, QMessageBox.Ok)
    #         else:
    #             QMessageBox.warning(self,"Input Warning","Please Select a Save Path before generate manual surface."
    #                                                     ,QMessageBox.Ok)

    def saveTxtFile(self):
        if self.UI.pixmap is not None:
            # self.refresh_Points()
            Mpicker_core_gui.Init_config_surface(self.UI.ini_config_path, self.UI.manual_config_path)
            np.savetxt(self.UI.ManualTxt_path,self.UI.tomo_manual_select,fmt = "%.2f", delimiter= "\t")

    def saveConfigFile(self):
        if self.UI.pixmap is not None:
            # create result folder
            chosed_label = self.scrollArea_manualsurf_vbox.itemAt(self.UI.tomo_manual_surfcurrent).layout().itemAt(0).widget()
            chosed_number = chosed_label.text().split(". ")[0]
            self.UI.result_folder = os.path.join(self.UI.Text_SavePath.toPlainText(), "manual_" + chosed_number + "_" +
                                                 os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
            if os.path.exists(self.UI.result_folder) == False:
                os.mkdir(self.UI.result_folder)
            self.UI.namesuffix = "manual_" + chosed_number
            self.UI.ManualTxt_path = os.path.join(self.UI.result_folder, self.UI.namesuffix + "_surf.txt")
            
            # save it to surf folder
            self.UI.manual_config_path = self.UI.ManualTxt_path.replace("_surf.txt",".config")
            if os.path.exists(self.UI.manual_config_path) is False:
                config_file = open(self.UI.manual_config_path, "w+")
            self.UI.manual_config.read(self.UI.manual_config_path, encoding='utf-8')
            # First Time Set up
            if self.UI.manual_config.has_section("Parameter") is False:
                self.UI.manual_config.add_section("Parameter")
                self.UI.manual_config.set('Parameter', "ID", chosed_number)
                self.UI.manual_config.set('Parameter', "txt_path",self.UI.ManualTxt_path)
            else:
                self.UI.manual_config.set("Parameter", "ID", chosed_number)
                self.UI.manual_config.set('Parameter', "txt_path", self.UI.ManualTxt_path)
            warnings.filterwarnings("ignore")
            with open(self.UI.manual_config_path,"w") as config_file:
                self.UI.manual_config.write(config_file)
            warnings.filterwarnings("default")

    def Reset_scrollArea_Surf_Select(self):
        # print("Reset_scrollArea_Surf_Select")
        if len(self.UI.tomo_manual_surf) > 0:
            # self.UI.label_Press_Flag = False
            Font = QFont("Agency FB", 9)
            for i in range(self.scrollArea_manualsurf_vbox.count()):
                chosed_label = self.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(0).widget()
                chosed_label.setStyleSheet("QLabel{color : black;}")
                if i == self.UI.tomo_manual_surfcurrent:
                    Font.setBold(True)
                else:
                    Font.setBold(False)
                if self.UI.resultpixmap is not None:
                    head = os.path.splitext(os.path.basename(self.UI.showResult_path))[0].split("_")[0]
                    if i == self.UI.tomo_check_current and head == "manual":
                        chosed_label.setStyleSheet("QLabel{color : orange;}")
                chosed_label.setFont(Font)

    def scrollArea_manualsurf_mousePressEvent(self,event):
        if self.UI.pixmap is not None:
            if event.buttons() & Qt.LeftButton:
                x = event.x()
                y = event.y()
                for i in range(self.scrollArea_manualsurf_vbox.count()):
                    # print("start searching ")
                    Font = QFont("Agency FB", 9)
                    chosed_label = self.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(0).widget()
                    pic_xmin, pic_ymin = chosed_label.x(), chosed_label.y()
                    pic_xmax, pic_ymax = chosed_label.x() + chosed_label.width(), chosed_label.y() + chosed_label.height()
                    if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                        self.UI.tomo_manual_surfcurrent = i
                        self.UI.tomo_manual_select = copy.deepcopy(
                            self.UI.tomo_manual_surf[self.UI.tomo_manual_surfcurrent])
                        break
                # print("self.UI.tomo_manual_surf = \n",self.UI.tomo_manual_surf)
                self.UI.tomo_manual_current = 0
                # self.ClearReference()
                self.Reset_scrollArea_Select()
                self.Reset_scrollArea_Surf_Select()
                # create result folder
                if len(self.UI.tomo_manual_surf) > 0:
                    chosed_label = self.scrollArea_manualsurf_vbox.itemAt(
                        self.UI.tomo_manual_surfcurrent).layout().itemAt(0).widget()
                    chosed_number = chosed_label.text().split(". ")[0]
                    self.UI.result_folder = os.path.join(self.UI.Text_SavePath.toPlainText(),
                                                         "manual_" + chosed_number + "_" +
                                                         os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
                    if os.path.exists(self.UI.result_folder):
                        self.UI.namesuffix = "manual_" + chosed_number
                        self.UI.ManualTxt_path = os.path.join(self.UI.result_folder, self.UI.namesuffix + "_surf.txt")
                        # print("self.UI.ManualTxt_path = ",self.UI.ManualTxt_path)
                    self.showMrcImage()
                    # print("self.UI.tomo_manual_number = ",self.UI.tomo_manual_number)
            elif event.buttons() & Qt.RightButton:
                self.UI.surf_right_pos = event.pos()

    def CheckBox_manual_Menu(self):
        self.Check_popMenu_manual   = QMenu()
        self.delete_select_action   = QAction(f'Delete Selected', self)
        self.show_select_action     = QAction(f'Show Selected', self)
        self.show_all_action        = QAction(f'Show All', self)
        self.close_all_action       = QAction(f'Hide All', self)
        if isinstance(self.delete_select_action, QAction):
            self.delete_select_action.triggered.connect(self.Delete_select_manual)
            self.show_select_action.triggered.connect(self.Show_selected_manual)
            self.show_all_action.triggered.connect(self.Show_all_manual)
            self.close_all_action.triggered.connect(self.Close_all_manual)
            self.Check_popMenu_manual.addAction(self.show_select_action)
            self.Check_popMenu_manual.addAction(self.show_all_action)
            self.Check_popMenu_manual.addAction(self.close_all_action)
            self.Check_popMenu_manual.addAction(self.delete_select_action)
            self.Check_popMenu_manual.move(QCursor.pos())
            self.Check_popMenu_manual.show()


    def Delete_select_manual(self):
        if self.UI.pixmap is not None:
            if self.UI.show_ask_check is False:
                self.Remove_manual_widget()
            else:
                self.scroll_remove_check_popupInfo()

    def scroll_remove_check_popupInfo(self):
        msg = QMessageBox()
        msg.setWindowTitle("Delete Warning")
        self.all_the_delete_names = ""
        first_flag = True
        for i in range(self.UI.scrollArea_manualsurf_vbox.count()):
            chosed_label = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(0).widget()
            chosed_checkbox = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(1).widget()
            if chosed_checkbox.isChecked():
                if first_flag:
                    self.all_the_delete_names += chosed_label.text()
                    first_flag = False
                else:
                    self.all_the_delete_names += "," + chosed_label.text()
        if self.all_the_delete_names != "":
            msg.setText(f'The following check Result will be deleted:\n'
                        f'[{self.all_the_delete_names}].\n'
                        f'And the Relevant Extract Result below(like 1-1,1-2...) will be deleted too! '
                        f'Do you still want to proceed? ')
        else:
            msg.setText(f'The following check Result will be deleted: '
                        f"[None].(You haven't chosen one to delete yet)\n"
                        f'And the Relevant Extract Result below(like 1-1,1-2...) will be deleted too! '
                        f'Do you still want to proceed? ')
        msg.setIcon(QMessageBox.Warning)
        cb = QCheckBox()
        cb.setText("Don't show this again")
        msg.setCheckBox(cb)
        msg.setWindowTitle("Delete Warning")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        final = msg.exec_()
        if final == QMessageBox.Yes:
            self.Remove_manual_widget()
            if cb.isChecked():
                self.UI.show_ask_check = False
        else:
            if cb.isChecked():
                self.UI.show_ask_check = False

    def Remove_manual_widget(self):
        for i in reversed(range(self.UI.scrollArea_manualsurf_vbox.count())):
            chosed_checkbox = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(1).widget()
            chosed_label = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(0).widget()
            num = chosed_label.text().split(".")[0]
            if chosed_checkbox.isChecked():
                del self.UI.tomo_manual_surf[i]
                self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(
                    1).widget().setParent(None)
                self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(
                    0).widget().setParent(None)
                self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(i).layout().setParent(None)
                if i < self.UI.tomo_manual_surfcurrent:
                    self.UI.tomo_manual_surfcurrent = len(self.UI.tomo_manual_surf) - 1
                    self.UI.tomo_check_current = self.UI.tomo_check_current - 1
                elif i > self.UI.tomo_manual_surfcurrent:
                    pass
                else:
                    self.UI.tomo_manual_surfcurrent = len(self.UI.tomo_manual_surf) - 1
                    self.UI.tomo_check_current      = 0
                    self.UI.tomo_manual_current     = 0
                    self.UI.tomo_manual_select      = []
                    self.UI.frame_manual.ClearReference()
                    self.UI.showMrcImage()
                    self.UI.frame_manual.Reset_scrollArea_Select()
                num = chosed_label.text().split(".")[0]
                for del_i in reversed(range(0, self.UI.scroll_hbox.count())):
                    chosed_label = self.UI.scroll_hbox.itemAt(del_i).layout()
                    self.UI.select_pic = chosed_label.itemAt(1).widget()
                    self.UI.select_label = chosed_label.itemAt(0).widget()
                    self.UI.select_path = chosed_label.itemAt(2).widget()
                    self.UI.select_layout = self.UI.scroll_hbox.itemAt(del_i)
                    if self.UI.select_label.text().split("-")[0][1:] == num:
                        #print("it will delete ", self.UI.select_label.text())
                        Remove_id = self.UI.select_label.text()
                        Remove_path = self.UI.select_path.text()
                        self.UI.select_pic.setParent(None)
                        self.UI.select_label.setParent(None)
                        self.UI.select_path.setParent(None)
                        self.UI.select_layout.setParent(None)
                        self.UI.Text_Information.clear()
                        if self.UI.showResult_path != self.UI.select_path.text():
                            pass
                        else:
                            for del_i in reversed(range(self.UI.select_result_scroll_vbox.count())):
                                self.UI.select_result_scroll_vbox.itemAt(del_i).widget().setParent(None)
                            self.UI.show_pic = None
                            self.UI.save_path = None
                            self.UI.resultpixmap = None
                            self.UI.sidepixmap = None
                            self.UI.tomo_result = None
                            self.UI.showResult_path = None
                            self.UI.graphicsView.Reset_scrollArea_Select()
                            self.UI.frame_manual.Reset_scrollArea_Surf_Select()
                            # self.tomo_extract = None
                            self.UI.graphicsScene_result.clear()
                            self.UI.graphicsScene_resultside.clear()
                            self.UI.graphicsScene_result = QGraphicsScene()
                            self.UI.graphicsScene_resultside = QGraphicsScene()
                        self.UI.select_label = None
                        self.UI.select_pic = None
                        self.UI.select_layout = None
                        self.UI.select_path = None
                delete_path = os.path.join(self.UI.Text_SavePath.toPlainText(), "manual_" + num + "_" +
                                           os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
                if os.path.exists(delete_path):
                    #print("delete ", delete_path)
                    self.UI.frame_Setting.remove_surface(delete_path)
        self.Show_selected_manual()

    def Show_selected_manual(self):
        self.Thread_Check_Selected_manual = QThread_Check_Selected_Manual()
        self.Thread_Check_Selected_manual.setParameters(self.UI)
        self.Thread_Check_Selected_manual.error.connect(self.show_all_manual_error)
        self.Thread_Check_Selected_manual.finished.connect(self.finished_show_all_manual)
        self.Thread_Check_Selected_manual.start()

    def show_all_manual_error(self):
        self.Thread_Check_Selected_manual.finished.disconnect()
        self.Thread_Check_Selected_manual.terminate()
        str_e = repr(self.Thread_Check_Selected_manual.exception)
        QMessageBox.warning(self, "Show Surf Error",
                            str_e, QMessageBox.Ok)

    def finished_show_all_manual(self):
        self.showMrcImage()

    def Show_all_manual(self):
        for i in reversed(range(self.UI.scrollArea_manualsurf_vbox.count())):
            chosed_checkbox = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(1).widget()
            chosed_checkbox.setChecked(True)
        self.Show_selected_manual()

    def Close_all_manual(self):
        for i in reversed(range(self.UI.scrollArea_manualsurf_vbox.count())):
            chosed_checkbox = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(1).widget()
            chosed_checkbox.setChecked(False)
        self.UI.tomo_manual_show = None
        self.showMrcImage()

class QThread_Check_Selected_Manual(QThread):
    def __init__(self):
        super(QThread_Check_Selected_Manual,self).__init__()
        self.exception = None

    error         = pyqtSignal(bool)
    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        try:
            self.UI.tomo_manual_show = []
            for i in reversed(range(self.UI.scrollArea_manualsurf_vbox.count())):
                chosed_checkbox = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(1).widget()
                if chosed_checkbox.isChecked():
                    self.UI.tomo_manual_show.append(copy.deepcopy(self.UI.tomo_manual_surf[i]))
        except Exception as e:
            self.exception = e
            self.error.emit(True)