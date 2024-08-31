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

from PyQt5.QtWidgets import     QGraphicsView,QApplication,QGraphicsScene,\
                                QMenu,QAction,QCheckBox,QMessageBox,\
                                QVBoxLayout,QWidget,QLabel,QShortcut
from PyQt5.QtGui import QPixmap,QImage,QCursor,QFont,QKeySequence,QTransform
from PyQt5.QtCore import QThread,Qt,pyqtSignal
from PIL import Image,ImageDraw,ImageEnhance
from scipy.ndimage import binary_dilation
import numpy as np
import mrcfile
import os,copy,json
from mpicker_item import Cross,Arrow,Circle
import configparser
from Mpicker_convert_mrc import read_surface_coord,coords2image,read_surface_mrc

for_fake_import = False
if for_fake_import:
    import Mpicker_gui


class Mpicker_MrcView(QGraphicsView):
    def __init__(self, *__args):
        super(Mpicker_MrcView, self).__init__(*__args)
        # Set short cut
        # Hot key conflict with Ctrl+S
        self.Shortcut_AddPoint = QShortcut(QKeySequence('Shift+S'), self)
        self.Shortcut_NextPoint = QShortcut(QKeySequence('Shift+X'), self)
        self.Shortcut_DeletePoint = QShortcut(QKeySequence('Shift+D'), self)

    def setParameters(self, UI):
        self.UI : Mpicker_gui.UI = UI
        self.UI.showMrcImage = self.showMrcImage
        self.UI.Button_Delete_auto.clicked.connect(self.DeleteReference)
        self.UI.Button_Clear_auto.clicked.connect(self.ClearReference)
        self.UI.Button_next_auto.clicked.connect(self.nextPoint)
        # self.UI.Button_Erase_auto.clicked.connect(self.ErasePoint)
        # change the parameter
        self.UI.comboBox_Surfmode.activated.connect(self.change_Surfmode)
        self.UI.comboBox_Direction.activated.connect(self.change_Surfmode)
        self.UI.comboBox_xyz.activated.connect(self.change_Surfmode)
        self.UI.comboBox_Nearerode.currentTextChanged.connect(self.change_Nearerode)
        # Select Points
        self.scrollArea_Select                          = self.UI.scrollArea_Select
        self.scrollArea_Select_vbox                     = self.UI.scrollArea_Select_vbox
        self.scrollArea_Select_widget                   = QWidget()
        self.scrollArea_Select_widget.mousePressEvent   = self.ScrollSelectpoint_mousePressEvent

        self.scrollArea_auto_Select                     = self.UI.scrollArea_auto_Select
        self.scrollArea_auto_Select_vbox                = self.UI.scrollArea_auto_Select_vbox
        self.scrollArea_auto_Select_widget              = QWidget()
        self.scrollArea_auto_Select_widget.mousePressEvent = self.scrollArea_auto_Select_mousePressEvent
        # Right click menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.GraphicViewMenu)
        self.scrollArea_Select_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.scrollArea_Select_widget.customContextMenuRequested.connect(self.CheckBox_Menu)
        # Cross Cursor
        # Gap and line width setting
        self.length = 5
        # self.width  = 2

        # Erase
        # self.UI.Button_Erase_manual.clicked.connect(self.ErasePoint)
        # self.EraseClickFlag = False
        # self.EraseFlag = False
        # self.transform = QTransform()
        # Set Cursor pixmap
        # self.EraseCursorlen = 30
        # earse_path = os.path.abspath(os.path.dirname(__file__))
        # earse_file = os.path.join(earse_path, 'Pic', 'EraseCursor.png')
        # pixmap = QPixmap(earse_file)
        # scaled_pixmap = pixmap.scaled(self.EraseCursorlen, self.EraseCursorlen)
        # self.cursorpix = QCursor(scaled_pixmap, -10, -10)

        self.Thread_Open_Boundary = QThread_Open_Boundary()
        self.Thread_Open_Boundary.setParameters(self.UI)
        self.Thread_Open_Boundary.error.connect(self.raiseerror)

        self.Thread_Open_Extract = QThread_Open_Extract()
        self.Thread_Open_Extract.setParameters(self.UI)
        self.Thread_Open_Extract.finished.connect(self.showMrcImage)

    def CheckBox_Menu(self):
        self.Check_popMenu      = QMenu()
        self.delete_select_action  = QAction(f'Delete Selected', self)
        self.show_select_action = QAction(f'Show Selected', self)
        self.show_all_action    = QAction(f'Show All',self)
        self.close_all_action   = QAction(f'Hide All', self)
        if isinstance(self.delete_select_action,QAction):
            self.delete_select_action.triggered.connect(self.Delete_select_surf)
            self.show_select_action.triggered.connect(self.Show_selected_surf)
            self.show_all_action.triggered.connect(self.Show_all_action)
            self.close_all_action.triggered.connect(self.Close_all_surf)
            self.Check_popMenu.addAction(self.show_select_action)
            self.Check_popMenu.addAction(self.show_all_action)
            self.Check_popMenu.addAction(self.close_all_action)
            self.Check_popMenu.addAction(self.delete_select_action)
            self.Check_popMenu.move(QCursor.pos())
            self.Check_popMenu.show()

    def Show_all_action(self):
        for i in range(self.UI.graphicsView.scrollArea_Select_vbox.count()):
            chosed_checkbox     = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(1).widget()
            chosed_checkbox.setChecked(True)
        self.Show_selected_surf()


    def Close_all_surf(self):
        self.UI.tomo_check_show = None
        for i in range(self.UI.graphicsView.scrollArea_Select_vbox.count()):
            chosed_checkbox     = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(1).widget()
            if chosed_checkbox.isChecked():
                chosed_checkbox.setChecked(False)
        self.showMrcImage()

    def Show_selected_surf(self):
        self.Thread_Check_All = QThread_Check_All()
        self.Thread_Check_All.setParameters(self.UI)
        self.Thread_Check_All.error.connect(self.show_all_surf_error)
        self.Thread_Check_All.finished.connect(self.finished_show_all_surf)
        self.Thread_Check_All.start()

    def show_all_surf_error(self):
        self.Thread_Check_All.finished.disconnect()
        self.Thread_Check_All.terminate()
        str_e = repr(self.Thread_Check_All.exception)
        QMessageBox.warning(self, "Show Surf Error",
                            str_e, QMessageBox.Ok)

    def finished_show_all_surf(self):
        # if np.all(self.UI.tomo_check_show == 0):
        if self.UI.tomo_check_show is None:
            return
        if len(self.UI.tomo_check_show) == 0:
            self.UI.tomo_check_show = None
        else:
            self.showMrcImage()

    def Delete_select_surf(self):
        if self.UI.pixmap is not None:
            if self.UI.show_ask_check is False:
                self.Remove_check_widget()
            else:
                self.scroll_remove_check_popupInfo()

    def scroll_remove_check_popupInfo(self):
        msg = QMessageBox()
        msg.setWindowTitle("Delete Warning")
        self.all_the_delete_names   = ""
        first_flag                  = True
        for i in range(self.UI.graphicsView.scrollArea_Select_vbox.count()):
            chosed_label        = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()
            chosed_checkbox     = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(1).widget()
            if chosed_checkbox.isChecked():
                if first_flag:
                    self.all_the_delete_names += chosed_label.text()
                    first_flag                 = False
                else:
                    self.all_the_delete_names += ","+chosed_label.text()
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
            self.Remove_check_widget()
            if cb.isChecked():
                self.UI.show_ask_check = False
        else:
            if cb.isChecked():
                self.UI.show_ask_check = False

    def Remove_check_widget(self):
        for i in reversed(range(self.UI.graphicsView.scrollArea_Select_vbox.count())):
            chosed_checkbox = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(1).widget()
            chosed_label    = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()
            num             = chosed_label.text().split(".")[0]
            if chosed_checkbox.isChecked():
                del self.UI.tomo_select_surf[i]
                del self.UI.tomo_select_surf_xyz[i]
                del self.UI.tomo_select_surf_mode[i]
                del self.UI.tomo_select_surf_direction[i]
                #del self.UI.minsurf[i]
                self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(
                    1).widget().setParent(None)
                self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(
                    0).widget().setParent(None)
                self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().setParent(None)
                if i < self.UI.tomo_surfcurrent:
                    self.UI.tomo_surfcurrent = len(self.UI.tomo_select_surf) - 1
                    self.UI.tomo_check_current = self.UI.tomo_check_current - 1
                elif i > self.UI.tomo_surfcurrent:
                    pass
                else:
                    self.UI.tomo_surfcurrent = len(self.UI.tomo_select_surf) - 1
                    self.UI.tomo_check_current = 0
                    self.UI.graphicsView.ClearReference()
                    # self.UI.graphicsView.Reset_scrollArea_Select()
                    self.UI.SURF_path = None
                    self.UI.tomo_extract = None
                    self.UI.showMrcImage()
                for del_i in reversed(range(0, self.UI.scroll_hbox.count())):
                    chosed_label = self.UI.scroll_hbox.itemAt(del_i).layout()
                    self.UI.select_pic = chosed_label.itemAt(1).widget()
                    self.UI.select_label = chosed_label.itemAt(0).widget()
                    self.UI.select_path = chosed_label.itemAt(2).widget()
                    self.UI.select_layout = self.UI.scroll_hbox.itemAt(del_i)
                    if self.UI.select_label.text().split("-")[0] == num:
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
                delete_path = os.path.join(self.UI.Text_SavePath.toPlainText(), "surface_" + num + "_" +
                                           os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
                if os.path.exists(delete_path):
                    # print("delete ",delete_path)
                    self.UI.frame_Setting.remove_surface(delete_path)
        self.Show_selected_surf()

    def save_Surf_parameter(self):
        if self.UI.pixmap is not None:
            if self.UI.tomo_current < len(self.UI.surf_mode):
                self.UI.surf_mode[self.UI.tomo_current]      = self.UI.comboBox_Surfmode.currentText()
                self.UI.surf_xyz[self.UI.tomo_current]       = self.UI.comboBox_xyz.currentText()
                self.UI.surf_direction[self.UI.tomo_current] = self.UI.comboBox_Direction.currentText()
            else:
                QMessageBox.warning(self, "Surf parameters Warning",
                                    "Surf Parameters out of bound"
                                    , QMessageBox.Ok)


    def change_Surfmode(self):
        if self.UI.pixmap is not None:
            if self.UI.comboBox_Surfmode.currentText() != "" \
                and self.UI.comboBox_xyz.currentText() != "" \
                and self.UI.comboBox_Direction.currentText() != "":
                self.save_Surf_parameter()
                self.showMrcImage()

    def showMrcImage(self, fromResult=False):
        if self.UI.tomo_show is not None and self.UI.allow_showMrcImage:
            if self.UI.realz <= self.UI.tomo_show.shape[0] and self.UI.realz >= 1:
                mrc_image_Image = self.UI.tomo_show[self.UI.realz-1,:,:]
                # mrc_image_data  = self.UI.tomo_data[self.UI.realz-1,:,:]
                # mrc_Image_sum   = np.sum(mrc_image_Image)
                # mrc_data_sum    = np.sum(mrc_image_data)
                # if self.UI.show_mrc_flag == True and mrc_Image_sum == mrc_data_sum:
                if self.UI.show_mrc_flag == True and self.UI.tomo_show is self.UI.tomo_data:
                    #print("test: min = ", self.UI.mrc_tomo_min,"||max = ",self.UI.mrc_tomo_max)
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
                if self.UI.tomo_check_show is not None and self.UI.Hide_Surf_Flag is False:
                    mrc_image_Image = np.array(mrc_image_Image) # shape (y,x,3)
                    # extract_image = self.UI.tomo_check_show[self.UI.realz - 1, :, :]
                    extract_image = coords2image(self.UI.tomo_check_show, mrc_image_Image.shape, self.UI.realz-1)
                    kernel = np.ones((2, 2), bool)
                    extract_image = binary_dilation(extract_image, structure=kernel)
                    mrc_image_Image[extract_image] = [127, 0, 255]
                    mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
                if self.UI.tomo_extract is not None and self.UI.Hide_Surf_Flag is False:
                    mrc_image_Image = np.array(mrc_image_Image)
                    # extract_image = self.UI.tomo_extract[self.UI.realz-1,:,:]
                    extract_image = coords2image(self.UI.tomo_extract, mrc_image_Image.shape, self.UI.realz-1)
                    kernel = np.ones((2, 2), bool)
                    extract_image = binary_dilation(extract_image,structure=kernel)
                    mrc_image_Image[extract_image] = [255,128,0]
                    # mrc_image_Image[extract_image>0] = [255,0,0]

                    # kernel3 = np.ones((5, 5), np.uint8)
                    # just for 8-14 ##change
                    # yelvti_dir='D:\mycode\mpicker\\tmp_mpicker\gui_out\emd_10780_corrected_show\surface_8_emd_10780_corrected'
                    # mrcl_path=os.path.join(yelvti_dir,'mask_8-15_z27.mrc')
                    # mrcm_path=os.path.join(yelvti_dir,'mask_8-15_z31.mrc')
                    # mrcr_path=os.path.join(yelvti_dir,'mask_8-15_z33.mrc')
                    # mrcr1_path=os.path.join(yelvti_dir,'mask_8-15_z39_1.mrc') # y 620 to 800
                    # mrcr2_path=os.path.join(yelvti_dir,'mask_8-15_z39_2.mrc') # y 240 530
                    # with mrcfile.open(mrcl_path) as mrc:
                    #     datal=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     # datal = binary_dilation(datal,structure=kernel)
                    # with mrcfile.open(mrcm_path) as mrc:
                    #     datam=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     # datam = binary_dilation(datam,structure=kernel)
                    # with mrcfile.open(mrcr_path) as mrc:
                    #     datar=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     # datar = binary_dilation(datar,structure=kernel)
                    # with mrcfile.open(mrcr1_path) as mrc:
                    #     datar1=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     # datar1 = binary_dilation(datar1,structure=kernel)
                    # with mrcfile.open(mrcr2_path) as mrc:
                    #     datar2=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     # datar2 = binary_dilation(datar2,structure=kernel)
                    # mrc_image_Image[datal>0] = [0,176,240]
                    # mrc_image_Image[datam>0] = [255,0,0]
                    # mrc_image_Image[datar>0] = [255,192,0]
                    # mrc_image_Image[datar1>0] = [0,0,255]
                    # mrc_image_Image[datar2>0] = [255,0,255]
                    # ##

                    # for 18-2
                    # lanzao_dir='D:\\mycode\mpicker\\tmp_mpicker\\gui_out\\emd_13771_corrected\\surface_18_emd_13771_corrected'
                    # mrcl_path=os.path.join(lanzao_dir,'mask_18-2_z01.mrc')
                    # mrcm_path=os.path.join(lanzao_dir,'mask_18-2_z21.mrc')
                    # mrcr_path=os.path.join(lanzao_dir,'mask_18-2_z61.mrc')
                    # mrcd_path=os.path.join(lanzao_dir,'mask_18-2_y001.mrc')
                    # mrcu_path=os.path.join(lanzao_dir,'mask_18-2_y648.mrc')
                    # with mrcfile.open(mrcl_path) as mrc:
                    #     datal=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     datal = binary_dilation(datal,structure=kernel3)
                    # with mrcfile.open(mrcm_path) as mrc:
                    #     datam=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     datam = binary_dilation(datam,structure=kernel)
                    # with mrcfile.open(mrcr_path) as mrc:
                    #     datar=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     datar = binary_dilation(datar,structure=kernel3)
                    # with mrcfile.open(mrcd_path) as mrc:
                    #     datad=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     datad = binary_dilation(datad,structure=kernel3)
                    # with mrcfile.open(mrcu_path) as mrc:
                    #     datau=mrc.data[self.UI.realz-1,::-1,:].astype(np.uint8)
                    #     datau = binary_dilation(datau,structure=kernel3)
                    # mrc_image_Image[datam>0] = [255,0,0]
                    # mrc_image_Image[datal>0] = [0,0,0]
                    # mrc_image_Image[datar>0] = [0,0,0]
                    # mrc_image_Image[datau>0] = [0,0,0]
                    # mrc_image_Image[datad>0] = [0,0,0]
                    
                    mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
                mrc_image_Image = np.asarray(mrc_image_Image)
                height, width, channels = mrc_image_Image.shape
                bytesPerLine            = channels * width
                qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.UI.pixmap = QPixmap.fromImage(qImg)
                self.UI.graphicsScene.clear()
                self.UI.graphicsScene.addPixmap(self.UI.pixmap)
                self.UI.graphicsView.setScene(self.UI.graphicsScene)

                # GraphicsView GraphicsItem select point && Cursor
                self.Reset_arrow_show()
                self.refresh_Points()
                self.refresh_Cursor(fromResult)

                # if self.UI.label_Press_Flag:
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


    def refresh_Cursor(self, fromResult=False):
        if fromResult and self.UI.saveoutofbound:
            return
        if self.UI.pixmap is not None:
            cross = Cross()
            cross.setPos(self.UI.realx - 1, self.UI.tomo_show.shape[1] - self.UI.realy)
            self.UI.graphicsScene.addItem(cross)

    def refresh_Points(self):
        if self.UI.pixmap is not None:
            self.select_group = self.UI.graphicsScene.createItemGroup([])
            for i in range(len(self.UI.tomo_select)):
                pos = self.UI.tomo_select[i]
                drawx = pos[0] - 1
                drawy = self.UI.tomo_show.shape[1] - pos[1]
                if pos[2] == self.UI.realz:
                    if self.UI.arrow_angle[i] == -1:
                        cross = Cross()
                        cross.setPos(drawx, drawy)
                        cross.setColor("green")
                        self.select_group.addToGroup(cross)
                    else:
                        arrow = Arrow()
                        arrow.setPos(drawx, drawy)
                        arrow.setRotation(self.UI.arrow_angle[i])
                        self.select_group.addToGroup(arrow)
                    circle = Circle()
                    circle.setR(self.UI.radius_surf)
                    circle.setPos(drawx, drawy)
                    circle.setColor("green")
                    self.select_group.addToGroup(circle)
                else:
                    if abs(pos[2]-self.UI.realz) < self.UI.radius_surf:
                        circle = Circle()
                        circle.setPos(drawx, drawy)
                        circle.setColor("green")
                        radius = self.UI.radius_surf - abs(pos[2] - self.UI.realz)
                        circle.setR(radius)
                        self.select_group.addToGroup(circle)
            self.select_group.update()


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.UI.pixmap is not None and self.UI.tomo_show is not None:
            self.UI.pressX = int(round(self.mapToScene(event.pos()).x()+1))
            self.UI.pressY = int(self.UI.tomo_show.shape[1] - round(self.mapToScene(event.pos()).y()))
            if event.button() & Qt.LeftButton:
                if self.UI.pressX > 0 and self.UI.pressX <= self.UI.tomo_show.shape[2] and self.UI.pressY > 0 \
                        and self.UI.pressY <= self.UI.tomo_show.shape[1]:
                    # coord y is reversed in 3Dmod!
                    self.UI.realx = self.UI.pressX
                    self.UI.realy = self.UI.pressY
                    # self.UI.label_Press_Flag = True
                    self.UI.showMrcImage()
            # elif event.button() & Qt.RightButton:
            #     if self.UI.pressX > 0 and self.UI.pressX < self.UI.tomo_show.shape[2] and self.UI.pressY > 0 \
            #             and self.UI.pressY < self.UI.tomo_show.shape[1]:
            #         self.EraseClickFlag = True
            #         # self.UI.label_Press_Flag = False
            # else:
            #     self.EraseClickFlag = False

    # def mouseReleaseEvent(self, event):
    #     super().mouseReleaseEvent(event)
    #     if self.UI.pixmap is not None and self.EraseFlag:
    #         if event.button() & Qt.RightButton:
    #             self.EraseClickFlag = False

    # def mouseMoveEvent(self, event):
    #     super().mouseMoveEvent(event)
    #     if self.UI.pixmap is not None and self.EraseFlag and self.EraseClickFlag:
    #         self.UI.pressX = int(round(self.mapToScene(event.pos()).x() + 1))
    #         self.UI.pressY = int(self.UI.tomo_show.shape[1] - round(self.mapToScene(event.pos()).y()))
    #         select_item = self.UI.graphicsScene.itemAt(self.mapToScene(event.pos()).x(),
    #                                                 self.mapToScene(event.pos()).y(), self.transform)
    #         if select_item is not None:
    #             if select_item.type() == 65536:
    #                 self.length = 10
    #                 if self.UI.tabWidget.currentWidget().objectName() == "tab_auto":
    #                     self.deleteReference()
    #                 elif self.UI.tabWidget.currentWidget().objectName() == "tab_manual":
    #                     self.UI.frame_manual.deleteReference()
    #                 else:
    #                     pass
    #                 self.length = 5

    def enterEvent(self, event):
        # if self.EraseFlag == False:
        #     QApplication.setOverrideCursor(Qt.CrossCursor)
        # else:
        #     QApplication.setOverrideCursor(self.cursorpix)
        super().enterEvent(event)
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
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(1.1,1.1)

    def zoomOut(self):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(1/1.1,1/1.1)

    def GraphicViewMenu(self):
        if self.UI.pixmap is not None and self.UI.tabWidget.currentWidget().objectName() == "tab_auto":
            self.groupBox_menu = QMenu(self)
            self.actionA = QAction(u'Save as Reference', self)
            self.actionA.setShortcut("Shift+S")
            self.groupBox_menu.addAction(self.actionA)
            self.actionA.triggered.connect(self.saveReference)
            Gap = self.length
            if len(self.UI.tomo_select) > 1:
                self.actionC = QAction(u'Next Point',self)
                self.actionC.setShortcut("Shift+X")
                self.groupBox_menu.addAction(self.actionC)
                self.actionC.triggered.connect(self.nextPoint)
            for select in self.UI.tomo_select:
                if self.UI.realz == select[2] and self.UI.pressX <= select[0]+Gap and \
                        self.UI.pressX >= select[0]-Gap and self.UI.pressY <= select[1] + Gap and self.UI.pressY > select[1] - Gap:
                    self.actionB = QAction(u'Delete Reference', self)
                    self.actionB.setShortcut("Shift+D")
                    self.groupBox_menu.addAction(self.actionB)
                    self.actionB.triggered.connect(self.deleteReference)
            self.actionD = QAction(u'Save screenshot', self)
            self.actionD.triggered.connect(lambda : self.UI.save_pixmap(self.UI.pixmap, "Raw_xy"))
            self.groupBox_menu.addAction(self.actionD)

            self.groupBox_menu.popup(QCursor.pos())

    def deleteReference(self):
        Gap = self.length
        for i in range(len(self.UI.tomo_select)):
            select = self.UI.tomo_select[i]
            if self.UI.realz == select[2] and self.UI.pressX <= select[0] + Gap and self.UI.pressX >= select[
                0] - Gap and self.UI.pressY <= select[1] + Gap and self.UI.pressY > select[1] - Gap:
                self.number = self.scrollArea_auto_Select_vbox.itemAt(i).widget().text().split(". ")[0]
                del self.UI.tomo_select[i]
                del self.UI.surf_xyz[i]
                del self.UI.surf_mode[i]
                del self.UI.surf_direction[i]
                if len(self.UI.tomo_select) > 0:
                    if i <= self.UI.tomo_current:
                        if self.UI.tomo_current != 0:
                            self.UI.tomo_current = self.UI.tomo_current - 1
                        else:
                            self.UI.tomo_current = 0
                    self.UI.comboBox_Surfmode.setCurrentIndex(
                        self.UI.comboBox_Surfmode.findText(self.UI.surf_mode[self.UI.tomo_current]))
                    self.UI.comboBox_xyz.setCurrentIndex(
                        self.UI.comboBox_xyz.findText(self.UI.surf_xyz[self.UI.tomo_current]))
                    self.UI.comboBox_Direction.setCurrentIndex(
                        self.UI.comboBox_Direction.findText(self.UI.surf_direction[self.UI.tomo_current]))
                    self.Reset_scrollArea_Select_points()
                else:
                    self.Reset_scrollArea_Select_points()
                break
        self.showMrcImage()

    def DeleteReference(self):
        if self.UI.pixmap is not None:# and self.draw_flag:
            if len(self.UI.tomo_select) >= 1:
                del self.UI.tomo_select[self.UI.tomo_current]
                del self.UI.surf_xyz[self.UI.tomo_current]
                del self.UI.surf_mode[self.UI.tomo_current]
                del self.UI.surf_direction[self.UI.tomo_current]
                if self.UI.tomo_current != 0:
                    self.UI.tomo_current = self.UI.tomo_current - 1
                    self.UI.realx = self.UI.tomo_select[self.UI.tomo_current][0]
                    self.UI.realy = self.UI.tomo_select[self.UI.tomo_current][1]
                    self.UI.realz = self.UI.tomo_select[self.UI.tomo_current][2]
                    self.UI.comboBox_Surfmode.setCurrentIndex(
                        self.UI.comboBox_Surfmode.findText(self.UI.surf_mode[self.UI.tomo_current]))
                    self.UI.comboBox_xyz.setCurrentIndex(
                        self.UI.comboBox_xyz.findText(self.UI.surf_xyz[self.UI.tomo_current]))
                    self.UI.comboBox_Direction.setCurrentIndex(
                        self.UI.comboBox_Direction.findText(self.UI.surf_direction[self.UI.tomo_current]))
                else:
                    if len(self.UI.tomo_select) > 0:
                        self.UI.tomo_current = len(self.UI.tomo_select) - 1
                        self.UI.realx = self.UI.tomo_select[self.UI.tomo_current][0]
                        self.UI.realy = self.UI.tomo_select[self.UI.tomo_current][1]
                        self.UI.realz = self.UI.tomo_select[self.UI.tomo_current][2]
                        self.UI.comboBox_Surfmode.setCurrentIndex(
                            self.UI.comboBox_Surfmode.findText(self.UI.surf_mode[self.UI.tomo_current]))
                        self.UI.comboBox_xyz.setCurrentIndex(
                            self.UI.comboBox_xyz.findText(self.UI.surf_xyz[self.UI.tomo_current]))
                        self.UI.comboBox_Direction.setCurrentIndex(
                            self.UI.comboBox_Direction.findText(self.UI.surf_direction[self.UI.tomo_current]))
                    else:
                        self.UI.tomo_current = 0
                        self.UI.realx = int(self.UI.tomo_show.shape[2] / 2)
                        self.UI.realy = self.UI.tomo_show.shape[1] - int(self.UI.tomo_show.shape[1] / 2)
                        self.UI.realz = int(self.UI.tomo_show.shape[0] / 2)
                self.Reset_scrollArea_Select_points()
                self.showMrcImage()


    def ClearReference(self):
        if self.UI.pixmap is not None:
            self.UI.tomo_current    = 0
            self.UI.tomo_select     = []
            self.UI.surf_xyz        = []
            self.UI.surf_mode       = []
            self.UI.surf_direction  = []
            for del_i in reversed(range(self.scrollArea_auto_Select_vbox.count())):
                self.scrollArea_auto_Select_vbox.itemAt(del_i).widget().setParent(None)
            self.showMrcImage()

    def remove_surf_config(self):
        self.UI.ini_config.read(self.UI.ini_config_path, encoding='utf-8')
        surface_all = ""
        for surface in self.UI.ini_config.get("Path","Surface").split(" "):
            surface_number = surface.split("_")[1]
            if surface_number != self.UI.number:
                surface_all = surface_all + " " + surface
        if surface_all == "":
            surface_all = "None"
        self.UI.ini_config.set("Path","Surface",surface_all)
        with open(self.UI.ini_config_path, "w") as config_file:
            self.UI.ini_config.write(config_file)

    def remove_extract_config(self):
        for i in reversed(range(self.UI.scroll_hbox.count())):
            if self.UI.scroll_hbox.itemAt(i) is not None:
                chosed_label = self.UI.scroll_hbox.itemAt(i)
                number = chosed_label.itemAt(0).widget().text().split("-")[0]
                make = chosed_label.itemAt(0).widget().text().split("-")[1]
                if number == self.UI.number:
                    self.UI.select_pic = self.UI.scroll_hbox.itemAt(i).layout().itemAt(1).widget()
                    self.UI.select_label = self.UI.scroll_hbox.itemAt(i).layout().itemAt(0).widget()
                    self.UI.select_path = self.UI.scroll_hbox.itemAt(i).layout().itemAt(2).widget()
                    self.UI.select_layout = self.UI.scroll_hbox.itemAt(i)
                    self.UI.scroll_remove()

    def saveReference(self):
        same_flag = 0
        for select in self.UI.tomo_select:
            if select[0] == self.UI.realx and select[1] == self.UI.realy and select[2] == self.UI.realz:
                same_flag = 1
                break
        if same_flag == 0:
            self.UI.tomo_select.append([self.UI.realx, self.UI.realy, self.UI.realz])
            self.UI.surf_mode.append(self.UI.comboBox_Surfmode.currentText())
            self.UI.surf_xyz.append(self.UI.comboBox_xyz.currentText())
            self.UI.surf_direction.append(self.UI.comboBox_Direction.currentText())
            self.UI.tomo_current = len(self.UI.tomo_select) - 1
            # Add a point
            self.Reset_scrollArea_Select_points()
            # self.scrollArea_auto_Select_widget.setLayout(self.scrollArea_auto_Select_vbox)
            # self.scrollArea_auto_Select.setWidget(self.scrollArea_auto_Select_widget)
            # self.UI.label_Press_Flag = False
            self.showMrcImage()

    def Reset_scrollArea_Select_points(self):
        # if self.UI.pixmap is not None:
        #print("Reset_scrollArea_Select_points")
        for del_i in reversed(range(self.scrollArea_auto_Select_vbox.count())):
            self.scrollArea_auto_Select_vbox.itemAt(del_i).widget().setParent(None)
        for select_i in range(len(self.UI.tomo_select)):
            select_label = QLabel()
            select_label.setText(
                f"{select_i + 1}. [{int(self.UI.tomo_select[select_i][0])},"
                f"{int(self.UI.tomo_select[select_i][1])},"
                f"{int(self.UI.tomo_select[select_i][2])}]")
            Font = QFont("Agency FB", 9)
            if select_i == self.UI.tomo_current:
                Font.setBold(True)
            select_label.setFont(Font)
            self.scrollArea_auto_Select_vbox.addWidget(select_label)
        self.scrollArea_auto_Select_widget.setLayout(self.scrollArea_auto_Select_vbox)
        self.scrollArea_auto_Select.setWidget(self.scrollArea_auto_Select_widget)

    def ReBold_scrollArea_Select_points(self):
        for select_i in range(self.scrollArea_auto_Select_vbox.count()):
            Font = QFont("Agency FB", 9)
            if select_i == self.UI.tomo_current:
                Font.setBold(True)
            self.scrollArea_auto_Select_vbox.itemAt(select_i).widget().setFont(Font)

    def nextPoint(self):
        if len(self.UI.tomo_select) > 1:
            if self.UI.tomo_current == len(self.UI.tomo_select) - 1:
                self.UI.tomo_current = 0
            else:
                self.UI.tomo_current += 1
            # show the pos
            self.Reset_scrollArea_Select_points()
            self.UI.realx = self.UI.tomo_select[self.UI.tomo_current][0]
            self.UI.realy = self.UI.tomo_select[self.UI.tomo_current][1]
            self.UI.realz = self.UI.tomo_select[self.UI.tomo_current][2]
            self.UI.comboBox_Surfmode.setCurrentIndex(
                self.UI.comboBox_Surfmode.findText(self.UI.surf_mode[self.UI.tomo_current]))
            self.UI.comboBox_xyz.setCurrentIndex(
                self.UI.comboBox_xyz.findText(self.UI.surf_xyz[self.UI.tomo_current]))
            self.UI.comboBox_Direction.setCurrentIndex(
                self.UI.comboBox_Direction.findText(self.UI.surf_direction[self.UI.tomo_current]))
            self.showMrcImage()

    def ScrollSelectpoint_mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.UI.allow_showMrcImage = False
            x = event.x()
            y = event.y()
            for i in range(self.scrollArea_Select_vbox.count()):
                Font = QFont("Agency FB", 9)
                chosed_label = self.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()#.widget()
                pic_xmin, pic_ymin = chosed_label.x(), chosed_label.y()
                pic_xmax, pic_ymax = chosed_label.x() + chosed_label.width(), chosed_label.y() + chosed_label.height()
                if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                    try:
                        self.UI.tomo_surfcurrent    = i #int(re.findall(r"(.*). ", chosed_label.text())[0]) - 1
                        self.UI.tomo_select         = copy.deepcopy(self.UI.tomo_select_surf[self.UI.tomo_surfcurrent])
                        self.UI.surf_xyz            = copy.deepcopy(self.UI.tomo_select_surf_xyz[self.UI.tomo_surfcurrent])
                        self.UI.surf_mode           = copy.deepcopy(self.UI.tomo_select_surf_mode[self.UI.tomo_surfcurrent])
                        self.UI.surf_direction      = copy.deepcopy(self.UI.tomo_select_surf_direction[self.UI.tomo_surfcurrent])
                        self.UI.tomo_current        = 0
                        # self.UI.realx = self.UI.tomo_select[self.UI.tomo_current][0]
                        # self.UI.realy = self.UI.tomo_select[self.UI.tomo_current][1]
                        # self.UI.realz = self.UI.tomo_select[self.UI.tomo_current][2]
                        
                        # self.UI.comboBox_Surfmode.setCurrentIndex(
                        #     self.UI.comboBox_Surfmode.findText(self.UI.surf_mode[self.UI.tomo_current]))
                        # self.UI.comboBox_xyz.setCurrentIndex(
                        #     self.UI.comboBox_xyz.findText(self.UI.surf_xyz[self.UI.tomo_current]))
                        # self.UI.comboBox_Direction.setCurrentIndex(
                        #     self.UI.comboBox_Direction.findText(self.UI.surf_direction[self.UI.tomo_current]))
                        # self.UI.advance_setting.spinBox_adminsurf.setValue(
                        #     self.UI.surf_config.getint('Parameter', 'minsurf'))
                        # if "ncpu" in self.UI.surf_config["Parameter"]:
                        #     self.UI.advance_setting.spinBox_adncpu.setValue(
                        #         self.UI.surf_config.getint('Parameter', 'ncpu'))
                        # else:
                        #     self.UI.advance_setting.spinBox_adncpu.setValue(4)

                        #except:
                            #self.UI.ad_minsurf = self.UI.surf_config.getint('Parameter', 'minsurf')
                        #self.UI.advance_setting.spinBox_adminsurf.setValue(self.UI.surf_config.getint('Parameter', 'minsurf'))
                        #self.UI.spinBox_minsurf.setValue(self.UI.minsurf[self.UI.tomo_surfcurrent])
                        self.Reset_scrollArea_Select()
                        self.Reset_scrollArea_Select_points()
                        break
                    except:
                        pass
            #self.refresh_Cursor()
            self.UI.allow_showMrcImage = True
            self.showMrcImage()
        elif event.buttons() & Qt.RightButton:
            self.UI.surf_right_pos = event.pos()
        else:
            pass

    def Reset_surf_show(self):
        if self.UI.pixmap is not None:
            #print("Reset_surf_show")
            chosed_label = self.UI.scrollArea_Select_vbox.itemAt(self.UI.tomo_surfcurrent).layout().itemAt(0).widget()#.widget()
            chosed_number = chosed_label.text().split(". ")[0]
            temp_result_folder  = os.path.join(self.UI.Text_SavePath.toPlainText(),
                                                 "surface_"+chosed_number+"_"+
                                              os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
            temp_config         = os.path.join(temp_result_folder,"surface_"+chosed_number+".config")
            temp_surf           = os.path.join(temp_result_folder,"surface_"+chosed_number+"_surf.mrc.npz")
            if os.path.exists(temp_result_folder) and temp_surf != self.UI.SURF_path:
                self.UI.result_folder = temp_result_folder
                self.UI.namesuffix = "surface_"+chosed_number
                self.UI.SURF_path  = os.path.join(self.UI.result_folder,self.UI.namesuffix+"_surf.mrc.npz")
                self.UI.surf_config.read(temp_config, encoding='utf-8')
                self.Reset_surf_show_parameter()

    def Reset_arrow_show(self):
        self.UI.arrow_angle = []
        #print("Reset_arrow_show")
        for i in range(len(self.UI.tomo_select)):
            if self.UI.surf_xyz[i] == "x":
                if self.UI.surf_direction[i] == "Left To Right":
                    self.UI.arrow_angle.append(0)
                else:
                    self.UI.arrow_angle.append(180)
            elif self.UI.surf_xyz[i] == "y":
                if self.UI.surf_direction[i] == "Left To Right":
                    self.UI.arrow_angle.append(270)
                else:
                    self.UI.arrow_angle.append(90)
            else:
                self.UI.arrow_angle.append(-1)

    def Reset_surf_show_parameter(self):
        if os.path.exists(self.UI.SURF_path) \
        or os.path.exists(self.UI.SURF_path.replace(".mrc.npz",".mrc")):
            #print("Reset_surf_show_parameter")
            if self.UI.tomo_current < len(self.UI.tomo_select):
                self.Thread_Open_Extract.start()
                self.UI.comboBox_Surfmode.setCurrentIndex(
                    self.UI.comboBox_Surfmode.findText(self.UI.surf_mode[self.UI.tomo_current]))
                self.UI.comboBox_xyz.setCurrentIndex(
                    self.UI.comboBox_xyz.findText(self.UI.surf_xyz[self.UI.tomo_current]))
                self.UI.comboBox_Direction.setCurrentIndex(
                    self.UI.comboBox_Direction.findText(self.UI.surf_direction[self.UI.tomo_current]))
                self.UI.comboBox_Nearerode.setCurrentIndex(
                    self.UI.comboBox_Nearerode.findText(self.UI.surf_config.get('Parameter', 'nearero')))
                self.UI.ad_minsurf = self.UI.surf_config.getint('Parameter','minsurf')
                self.UI.advance_setting.spinBox_adminsurf.setValue(
                    self.UI.surf_config.getint('Parameter', 'minsurf'))
                if "ncpu" in self.UI.surf_config["Parameter"]:
                    self.UI.advance_setting.spinBox_adncpu.setValue(
                        self.UI.surf_config.getint('Parameter', 'ncpu'))
                else:
                    self.UI.advance_setting.spinBox_adncpu.setValue(1)
                # except:
                # self.UI.ad_minsurf = self.UI.surf_config.getint('Parameter', 'minsurf')
                #self.UI.advance_setting.spinBox_adminsurf.setValue(self.UI.surf_config.getint('Parameter','minsurf'))
                #self.UI.spinBox_minsurf.setValue(self.UI.minsurf[self.UI.tomo_surfcurrent])
                Boundary_path = os.path.join(self.UI.Text_SavePath.toPlainText(),
                                                     'my_boundary_' + self.UI.comboBox_Nearerode.currentText() + '.mrc')
                if Boundary_path != self.UI.Boundary_path:
                    self.UI.Boundary_path = Boundary_path
                    self.Thread_Open_Boundary.start()
                if "maxpixel" in self.UI.surf_config["Parameter"]:
                    self.UI.spinBox_maxpixel.setValue(self.UI.surf_config.getint('Parameter', 'maxpixel'))
                else:
                    self.UI.spinBox_maxpixel.setValue(200)
                # self.UI.graphicsView.Reset_arrow_show()
                #print("Surf Path = ",self.UI.SURF_path)

                # with mrcfile.open(self.UI.SURF_path) as mrc:
                #     self.UI.tomo_extract = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.UI.tomo_extract)
                    # tomo_max = np.max(self.UI.tomo_extract)
                    # self.UI.tomo_extract = ((self.UI.tomo_extract - tomo_min) / (tomo_max - tomo_min) * 255).astype(
                    #             np.uint8)

    def Load_config_surface(self):
        self.UI.ini_config.read(self.UI.ini_config_path, encoding='utf-8')
        Surface_string  = self.UI.ini_config.get('Path','Surface')
        if Surface_string != "None":
            Surface_list    = Surface_string.split(' ')
            Surface_auto_list   = []
            Surface_manual_list = []
            for Surface_list_i in Surface_list:
                if Surface_list_i.split("_")[0] == "surface":
                    Surface_auto_list.append(Surface_list_i)
                elif Surface_list_i.split("_")[0] == "manual":
                    Surface_manual_list.append(Surface_list_i)
                else:
                    pass
            Surface_auto_list.sort(key=lambda x: int(x.split("_")[1]))
            Surface_manual_list.sort(key=lambda x: int(x.split("_")[1]))
            if len(Surface_auto_list) > 0:
                for Surface_list_i in Surface_auto_list:
                    self.UI.result_folder = os.path.join(os.path.abspath(self.UI.Text_SavePath.toPlainText()),Surface_list_i)
                    self.UI.surf_config_path  = os.path.join(self.UI.result_folder,Surface_list_i.split("_")[0]+"_"
                                                                 +Surface_list_i.split("_")[1]+".config")
                    self.UI.surf_config.read(self.UI.surf_config_path, encoding='utf-8')
                    Surface_number = self.UI.surf_config.getint("Parameter", "ID")
                    self.UI.tomo_select         = json.loads(self.UI.surf_config.get("Parameter","Points"))
                    if self.UI.surf_config.get("Parameter", "Facexyz")[0] == "[":
                        self.UI.surf_mode = json.loads(self.UI.surf_config.get("Parameter", "mode").replace('\'', '\"'))
                        self.UI.surf_xyz = json.loads(
                            self.UI.surf_config.get("Parameter", "Facexyz").replace('\'', '\"'))
                        self.UI.surf_direction = json.loads(
                            self.UI.surf_config.get("Parameter", "DirectionL2R").replace('\'', '\"'))
                    else:
                        self.UI.surf_mode = [self.UI.surf_config.get("Parameter", "mode")]
                        self.UI.surf_xyz = [self.UI.surf_config.get("Parameter", "Facexyz")]
                        self.UI.surf_direction = [self.UI.surf_config.get("Parameter", "DirectionL2R")]
                    # rank
                    # self.UI.surf_xyz            = json.loads(self.UI.surf_config.get("Parameter","mode").replace('\'', '\"'))
                    # self.UI.surf_mode           = json.loads(self.UI.surf_config.get("Parameter","Facexyz").replace('\'', '\"'))
                    # self.UI.surf_direction      = json.loads(self.UI.surf_config.get("Parameter","DirectionL2R").replace('\'', '\"'))
                    if "minsurf" in self.UI.surf_config["Parameter"]:
                        self.UI.minsurf.append(self.UI.surf_config.getint("Parameter", "minsurf"))
                    else:
                        self.UI.minsurf.append(10)
                    self.UI.tomo_select_surf.append(copy.deepcopy(self.UI.tomo_select))
                    self.UI.tomo_select_surf_xyz.append(copy.deepcopy(self.UI.surf_xyz))
                    self.UI.tomo_select_surf_mode.append(copy.deepcopy(self.UI.surf_mode))
                    self.UI.tomo_select_surf_direction.append(copy.deepcopy(self.UI.surf_direction))
                    self.UI.tomo_current = 0
                    self.UI.tomo_surfcurrent = len(self.UI.tomo_select_surf) - 1
                    # Add a point
                    self.UI.frame_Setting.config_set_select_points(Surface_number)
                    # select_label = QLabel()
                    # select_label.setText(
                    #         f"{Surface_number}. Surface")
                    # self.scrollArea_Select_vbox.addWidget(select_label)
                    # self.scrollArea_Select_widget.setLayout(self.scrollArea_Select_vbox)
                    # self.scrollArea_Select.setWidget(self.scrollArea_Select_widget)
                    self.UI.tomo_number = Surface_number + 1
                    self.UI.SURF_path = self.UI.surf_config_path.replace(".config","_surf.mrc.npz")
                    # self.UI.Picked_point =  self.UI.tomo_select[self.UI.tomo_current]
                self.Reset_scrollArea_Select()
                self.Reset_scrollArea_Select_points()
                #self.Reset_surf_show_parameter()
                self.UI.Init_MrcImage()
            for Surface_list_i in Surface_manual_list:
                self.UI.result_folder = os.path.join(os.path.abspath(self.UI.Text_SavePath.toPlainText()),
                                                     Surface_list_i)
                self.UI.manual_config_path = os.path.join(self.UI.result_folder, Surface_list_i.split("_")[0] + "_"
                                                        + Surface_list_i.split("_")[1] + ".config")
                self.UI.manual_config.read(self.UI.manual_config_path, encoding='utf-8')
                Surface_number = self.UI.manual_config.getint("Parameter", "ID")
                Surface_points = self.UI.manual_config_path.replace(".config", "_surf.txt")
                #Surface_points = self.UI.manual_config.get("Parameter","txt_path")
                self.UI.tomo_manual_select = np.loadtxt(Surface_points).astype(np.uint32)
                if self.UI.tomo_manual_select.ndim == 1:
                    self.UI.tomo_manual_select = self.UI.tomo_manual_select.tolist()
                    self.UI.tomo_manual_select = [self.UI.tomo_manual_select]
                else:
                    self.UI.tomo_manual_select = self.UI.tomo_manual_select.tolist()
                self.UI.tomo_manual_surf.append(copy.deepcopy(self.UI.tomo_manual_select))
                self.UI.tomo_manual_surfcurrent = len(self.UI.tomo_manual_surf) - 1
                self.UI.tomo_manual_current =  0
                self.UI.ManualTxt_path = self.UI.manual_config_path.replace(".config","_surf.txt")
                # Add a point
                self.UI.tomo_manual_number = Surface_number
                self.UI.frame_Setting.config_set_select_manual(False)
                # select_label = QLabel()
                # select_label.setText(f"{Surface_number}. Surface")
                # self.UI.scrollArea_manualsurf_vbox.addWidget(select_label)
                # self.UI.frame_manual.scrollArea_manualsurf_widget.setLayout(self.UI.scrollArea_manualsurf_vbox)
                # self.UI.frame_manual.scrollArea_manualsurf.setWidget(self.UI.frame_manual.scrollArea_manualsurf_widget)
                self.UI.tomo_manual_number = Surface_number + 1
            self.UI.frame_manual.Reset_scrollArea_Surf_Select()
            self.UI.frame_manual.Reset_scrollArea_Select()
            self.UI.frame_manual.scrollArea_manualpoints_widget.setLayout(self.UI.frame_manual.scrollArea_manualpoints_vbox)
            self.UI.frame_manual.scrollArea_manualpoints.setWidget(self.UI.frame_manual.scrollArea_manualpoints_widget)
            self.UI.Init_MrcImage()


    # def ErasePoint(self):
    #     if self.EraseFlag == False:
    #         if self.UI.Hide_Surf_Flag:
    #             QMessageBox.information(self, 'Reminder', 'You still hide your selected points\nPlease show them before you erase.',QMessageBox.Ok)
    #         self.EraseFlag = True
    #         self.customContextMenuRequested.disconnect()
    #         if self.UI.tabWidget.currentWidget().objectName() == "tab_auto":
    #             self.UI.Button_Erase_auto.setStyleSheet("background-color: gray")
    #         elif self.UI.tabWidget.currentWidget().objectName() == "tab_manual":
    #             self.UI.Button_Erase_manual.setStyleSheet("background-color: gray")
    #         else:
    #             pass
    #     else:
    #         self.EraseFlag = False
    #         if self.UI.tabWidget.currentWidget().objectName() == "tab_auto":
    #             self.customContextMenuRequested.connect(self.GraphicViewMenu)
    #             self.UI.Button_Erase_auto.setStyleSheet("background-color: None")
    #         elif self.UI.tabWidget.currentWidget().objectName() == "tab_manual":
    #             self.customContextMenuRequested.connect(self.UI.frame_manual.GraphicViewMenu)
    #             self.UI.Button_Erase_manual.setStyleSheet("background-color: None")
    #         else:
    #             pass

    def saveSurfReference(self):
        self.UI.tomo_select_surf.append(copy.deepcopy(self.UI.tomo_select))
        self.UI.tomo_select_surf_xyz.append(copy.deepcopy(self.UI.surf_xyz))
        self.UI.tomo_select_surf_mode.append(copy.deepcopy(self.UI.surf_mode))
        self.UI.tomo_select_surf_direction.append(copy.deepcopy(self.UI.surf_direction))
        #self.UI.minsurf.append(self.UI.ad_minsurf)
        self.UI.tomo_surfcurrent    = len(self.UI.tomo_select_surf) - 1
        self.UI.frame_Setting.config_set_select_points(self.UI.tomo_number)
        # select_label = QLabel()
        # select_label.setContextMenuPolicy(Qt.CustomContextMenu)
        # select_label.customContextMenuRequested.connect(self.UI.frame_Setting.Qlabel_Menu)
        # select_label.setText(
        #     f"{self.UI.tomo_number}. Surface")
        # self.scrollArea_Select_vbox.addWidget(select_label)
        # self.scrollArea_Select_widget.setLayout(self.scrollArea_Select_vbox)
        # self.scrollArea_Select.setWidget(self.scrollArea_Select_widget)
        self.Reset_scrollArea_Select()
        self.UI.tomo_number = self.UI.tomo_number + 1
        self.showMrcImage()

    def Reset_scrollArea_Select(self):
        if len(self.UI.tomo_select_surf) > 0:
            self.Reset_surf_show()
            #print("Reset_scroll_Area_Select")
            # self.UI.label_Press_Flag = False
            Font = QFont("Agency FB", 9)
            for i in range(self.scrollArea_Select_vbox.count()):
                chosed_label = self.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()#.widget()
                chosed_label.setStyleSheet("QLabel{color : black;}")
                if i == self.UI.tomo_surfcurrent:
                    Font.setBold(True)
                else:
                    Font.setBold(False)
                if self.UI.resultpixmap is not None:
                    head = os.path.splitext(os.path.basename(self.UI.showResult_path))[0].split("_")[0]
                    if i == self.UI.tomo_check_current and head != "manual":
                        chosed_label.setStyleSheet("QLabel{color : orange;}")
                chosed_label.setFont(Font)

    def scrollArea_auto_Select_mousePressEvent(self,event):
        if event.buttons() & Qt.LeftButton:
            x = event.x()
            y = event.y()
            for i in range(self.scrollArea_auto_Select_vbox.count()):
                chosed_label = self.scrollArea_auto_Select_vbox.itemAt(i).widget()
                pic_xmin, pic_ymin = chosed_label.x(), chosed_label.y()
                pic_xmax, pic_ymax = chosed_label.x() + chosed_label.width(), chosed_label.y() + chosed_label.height()
                if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                    try:
                        self.UI.tomo_current = i
                        self.UI.realx = self.UI.tomo_select[self.UI.tomo_current][0]
                        self.UI.realy = self.UI.tomo_select[self.UI.tomo_current][1]
                        self.UI.realz = self.UI.tomo_select[self.UI.tomo_current][2]
                        self.UI.comboBox_Surfmode.setCurrentIndex(
                            self.UI.comboBox_Surfmode.findText(self.UI.surf_mode[self.UI.tomo_current]))
                        self.UI.comboBox_xyz.setCurrentIndex(
                            self.UI.comboBox_xyz.findText(self.UI.surf_xyz[self.UI.tomo_current]))
                        self.UI.comboBox_Direction.setCurrentIndex(
                            self.UI.comboBox_Direction.findText(self.UI.surf_direction[self.UI.tomo_current]))
                        break
                    except:
                        pass
            self.ReBold_scrollArea_Select_points()
            self.showMrcImage()

    def raiseerror(self,bool):
        if bool == True:
            if bool:
                str_e = repr(self.Thread_Open_Boundary.exception)
                QMessageBox.warning(self, "Open Boundary Error", str_e+"\nboundary file might be deleted", QMessageBox.Ok)
                self.Thread_Open_Boundary.terminate()

    def change_Nearerode(self):
        if self.UI.pixmap is not None:
            Boundary_path = os.path.join(self.UI.Text_SavePath.toPlainText(),
                                         'my_boundary_' + self.UI.comboBox_Nearerode.currentText() + '.mrc')
            if os.path.exists(Boundary_path) and Boundary_path != self.UI.Boundary_path:
                self.UI.Boundary_path   = Boundary_path
                self.Nearerode_Thread_Open_Boundary = QThread_Open_Boundary()
                self.Nearerode_Thread_Open_Boundary.setParameters(self.UI)
                self.Nearerode_Thread_Open_Boundary.error.connect(self.raiseerror)
                # if np.array_equal(self.UI.tomo_show,self.UI.tomo_boundary):
                if self.UI.tomo_show is self.UI.tomo_boundary:
                    self.UI.show_mrc_flag = False
                    self.Nearerode_Thread_Open_Boundary.finished.connect(self.Thread_Open_Boundary_Show)
                # elif np.array_equal(self.UI.tomo_show,self.UI.tomo_mask):
                elif self.UI.tomo_show is self.UI.tomo_mask:
                    self.UI.show_mrc_flag = False
                    self.Nearerode_Thread_Open_Boundary.finished.connect(self.showMrcImage)
                else:
                    self.UI.show_mrc_flag = True
                    self.Nearerode_Thread_Open_Boundary.finished.connect(self.showMrcImage)
                self.Nearerode_Thread_Open_Boundary.start()


    def Thread_Open_Boundary_Show(self):
        self.UI.tomo_show = self.UI.tomo_boundary
        self.showMrcImage()

    # def Thread_Open_Boundary_Show_Raw(self):
    #     self.UI.tomo_show = self.UI.tomo_data
    #     self.showMrcImage()
    #
    # def Thread_Open_Mask_Show(self):
    #     self.UI.tomo_show = self.UI.tomo_mask
    #     self.showMrcImage()


class QThread_Check_All(QThread):
    def __init__(self):
        super(QThread_Check_All,self).__init__()
        self.exception = None

    error         = pyqtSignal(bool)
    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        try:
            # self.UI.tomo_check_show = np.zeros(self.UI.tomo_show.shape, dtype=bool)
            tomo_check_show = []
            tomo_check_show_mask = None
            for i in reversed(range(self.UI.graphicsView.scrollArea_Select_vbox.count())):
                chosed_checkbox = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(1).widget()
                chosed_label = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()
                num = chosed_label.text().split(".")[0]
                if chosed_checkbox.isChecked():
                    chosed_folder = os.path.join(self.UI.Text_SavePath.toPlainText(),
                                                 "surface_" + num + "_" +
                                                 os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
                    if os.path.exists(chosed_folder):
                        chosed_namesuffix = "surface_" + num
                        chosed_surf_path = os.path.join(chosed_folder, chosed_namesuffix + "_surf.mrc.npz")
                        if os.path.exists(chosed_surf_path):
                            # chosed_surf_data = read_surface_mrc(chosed_surf_path, dtype=bool)[:, ::-1, :]
                            # self.UI.tomo_check_show[chosed_surf_data] = 1
                            chosed_surf_data = read_surface_coord(chosed_surf_path)
                            if len(chosed_surf_data) > 0:
                                tomo_check_show.append(chosed_surf_data)
                        elif os.path.exists(chosed_surf_path.replace(".mrc.npz",".mrc")):
                            if tomo_check_show_mask is None: 
                                tomo_check_show_mask = np.zeros(self.UI.tomo_show.shape, dtype=bool)
                            chosed_surf_data = read_surface_mrc(chosed_surf_path, dtype=bool)
                            tomo_check_show_mask[chosed_surf_data] = 1
            if tomo_check_show_mask is not None:
                chosed_surf_data = np.argwhere(tomo_check_show_mask).astype(int)
                if len(chosed_surf_data) > 0:
                    tomo_check_show.append(chosed_surf_data)
            if len(tomo_check_show) > 0:
                self.UI.tomo_check_show = np.concatenate(tomo_check_show, axis=0)
        except Exception as e:
            self.UI.tomo_check_show = None
            self.exception = e
            self.error.emit(True)

class QThread_Open_Boundary(QThread):
    def __init__(self):
        super(QThread_Open_Boundary,self).__init__()
        self.exception = None

    error         = pyqtSignal(bool)

    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        try:
            with mrcfile.mmap(self.UI.Boundary_path) as mrc:
                self.UI.tomo_boundary = mrc.data[:, ::-1, :]
                # tomo_min = np.min(self.UI.tomo_boundary)
                # tomo_max = np.max(self.UI.tomo_boundary)
                # if tomo_max != tomo_min:
                #     self.UI.tomo_boundary = (
                #             (self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255)#.astype(np.uint8)
                # else:
                #     self.UI.tomo_boundary = np.zeros(self.UI.tomo_boundary.shape)#.astype(np.uint8)
                #self.UI.show_mrc_flag = False
                #self.UI.tomo_boundary = self.UI.tomo_boundary.astype(np.uint8)
                # self.UI.tomo_boundary = ((self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(
                #     np.uint8)
        except Exception as e:
            self.exception = e
            self.error.emit(True)


class QThread_Open_Extract(QThread):
    def __init__(self):
        super(QThread_Open_Extract,self).__init__()

    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        # self.UI.tomo_extract = read_surface_mrc(self.UI.SURF_path, dtype=bool)[:, ::-1, :]
        self.UI.tomo_extract = read_surface_coord(self.UI.SURF_path)

