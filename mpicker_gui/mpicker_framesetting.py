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
from PyQt5.QtCore import QThread,Qt,pyqtSignal
from PyQt5.QtGui import QCursor
from PIL import Image,ImageDraw,ImageEnhance
import numpy as np
import mrcfile
import os,configparser,copy,json
from mpicker_core import Get_Boundary
from Mpicker_convert_mrc import read_surface_coord
from PyQt5.QtGui import QPixmap,QImage,QFont,QKeySequence
from PyQt5.QtCore import QWaitCondition,QMutex
from mpicker_framememseg import UI_Mpicker_FrameMemseg
from mpicker_classeditor import class_File
for_fake_import = False
if for_fake_import:
    import Mpicker_gui

class Mpicker_FrameSetting(QFrame):
    def __init__(self, *__args):
        super(Mpicker_FrameSetting, self).__init__(*__args)
        self.delete_action          = None
        self.delete_action_manual   = None
        self.Surf_popMenu           = None
        self.delete_manual          = None
        self.delete_surf            = None
        self.Switch_menu = QMenu()
        self.actionRaw = QAction("Raw", self)
        self.Switch_menu.addAction(self.actionRaw)
        self.actionRaw.triggered.connect(self.showRaw)
        self.actionRaw.setDisabled(True)
        self.actionMask = QAction("Mask", self)
        self.Switch_menu.addAction(self.actionMask)
        self.actionMask.triggered.connect(self.showMask)
        self.actionMask.setDisabled(True)
        self.actionBoundary = QAction("Boundary", self)
        self.Switch_menu.addAction(self.actionBoundary)
        self.actionBoundary.triggered.connect(self.showBoundary)
        self.actionBoundary.setDisabled(True)

        self.Mask_menu = QMenu()
        self.actionOpen = QAction("Open Mask", self)
        self.Mask_menu.addAction(self.actionOpen)
        self.actionOpen.triggered.connect(self.clicker_Openmask)
        self.actionGet = QAction("Get Mask", self)
        self.Mask_menu.addAction(self.actionGet)
        self.actionGet.triggered.connect(self.clicker_Getmask)

        self.Thread_Openconfig = QThread_OpenConfig()
        self.Thread_Openconfig.finished.connect(self.Reset_config_Button)
        self.Thread_Openconfig.set_raw_enabled.connect(self.config_set_raw_enabled)
        self.Thread_Openconfig.set_mask_enabled.connect(self.config_set_mask_enabled)
        self.Thread_Openconfig.set_boundary_enabled.connect(self.config_set_boundary_enabled)
        self.Thread_Openconfig.set_save_path_text.connect(self.config_set_save_path_text)
        self.Thread_Openconfig.set_select_points.connect(self.config_set_select_points)
        self.Thread_Openconfig.set_select_manual.connect(self.config_set_select_manual)
        self.Thread_Openconfig.set_select_manual_points.connect(self.config_set_select_manual_points)
        self.Thread_Openconfig.set_scroll_surf.connect(self.config_set_scroll_surf)
        self.Thread_Openconfig.set_extract_surf.connect(self.config_set_extract_surf)
        self.Thread_Openconfig.load_progress.connect(self.config_load_progress)


    def setParameters(self, UI):
        self.UI : Mpicker_gui.UI = UI
        # FrameSetting Function
        self.Thread_FinishedOpen = QThread_FinishedOpen()
        self.Thread_FinishedOpen.set_config_all.connect(self.UI.Init_config_all)
        self.Thread_FinishedOpen.set_surface.connect(self.UI.graphicsView.Load_config_surface)
        self.Thread_FinishedOpen.set_find.connect(self.UI.Load_config_find)
        self.Thread_FinishedOpen.error.connect(self.finishedopenError)
        # Buttons in Frame Setting
        #self.UI.Button_Openmask.clicked.connect(self.clicker_Openmask)
        self.UI.Button_Openraw.clicked.connect(self.clicker_openRaw)
        self.UI.Button_Hidesurf.clicked.connect(self.clicker_HideSurface)
        self.UI.Button_Selectsavepath.clicked.connect(self.clicker_savePath)
        # Contrast in Frame Setting
        self.UI.horizontalSlider_Contrast.valueChanged.connect(self.slide_Contrast)
        self.UI.horizontalSlider_Bright.valueChanged.connect(self.slide_Bright)
        self.UI.doubleSpinBox_Contrast.valueChanged.connect(self.doubleSpinBox_L_Contrast)
        self.UI.doubleSpinBox_Bright.valueChanged.connect(self.doubleSpinBox_L_Bright)
        # XYZ in Frame Setting
        self.UI.doubleSpinBox_X.valueChanged.connect(self.SpinBox_X)
        self.UI.doubleSpinBox_Y.valueChanged.connect(self.SpinBox_Y)
        self.UI.doubleSpinBox_Z.valueChanged.connect(self.SpinBox_Z)
        self.UI.horizontalSlider_z.valueChanged.connect(self.slide_z)
        self.UI.spinBox_z.valueChanged.connect(self.SpinBox_z)
        # Menu Setting of Switch show
        self.UI.Button_Switchshow.setMenu(self.Switch_menu)
        self.UI.Button_Openmask.setMenu(self.Mask_menu)
        self.UI.Button_OpenSession.clicked.connect(self.Open_config)



    def Open_config(self):
        fname = QFileDialog.getOpenFileName(self, "Open Config File",
                                            os.getcwd(),"Config Files (*.config);;All Files (*)")
        if fname[0] != '':
            try:
                self.clear_all_widgets()
                self.UI.ini_config_path = fname[0]
                self.UI.ini_config.read(self.UI.ini_config_path, encoding='utf-8')
                self.Thread_Openconfig.setParameters(self.UI)
                self.Thread_Openconfig.start()
                self.disabled_all_button()
            except Exception as e:
                str_e = repr(e)
                QMessageBox.warning(self,"Input Warning",str_e,QMessageBox.Ok)

    def clear_all_widgets(self):
        self.UI.show_ask        = True
        self.UI.show_ask_surf   = True
        self.UI.show_ask_check  = True
        self.UI.MRC_path        = None
        self.UI.MASK_path       = None
        self.UI.Boundary_path   = None
        self.UI.pixmap          = None
        self.UI.realx           = 1
        self.UI.realy           = 1
        self.UI.realz           = 1
        self.UI.tomo_check_show = None
        self.UI.tomo_manual_show= None
        self.UI.tomo_data       = None
        self.UI.tomo_mask       = None
        self.UI.tomo_boundary   = None
        self.UI.tomo_show       = None
        self.UI.tomo_select     = []
        self.UI.tomo_select_surf= []
        self.UI.tomo_select_surf_xyz = []
        self.UI.tomo_select_surf_mode = []
        self.UI.tomo_select_surf_direction= []
        #self.UI.minsurf         = []
        self.UI.tomo_surfcurrent= 0
        self.UI.tomo_current    = 0
        self.UI.tomo_manual_surf = []
        self.UI.mrc_tomo_max    = 255
        self.UI.mrc_tomo_min    = 0
        self.UI.old_mrc_tomo_max = 255
        self.UI.old_mrc_tomo_min = 0
        self.UI.tomo_manual_select = []
        self.UI.tomo_manual_current = 0
        self.UI.tomo_manual_surfcurrent = 0
        self.UI.tomo_manual_number = 1
        self.UI.tomo_number     = 1
        self.UI.tomo_show_std   = 0
        self.UI.show_pic        = None
        self.UI.save_path       = None
        self.UI.tomo_result     = None
        self.UI.tomo_addresult  = None
        self.UI.resultpixmap    = None
        self.UI.resultx         = 1
        self.UI.resulty         = 1
        self.UI.resultz         = 1
        self.UI.label_resultx   = 0
        self.UI.label_resulty   = 0
        self.UI.Hide_Surf_Flag  = False
        self.UI.Picked_point    = [-1, -1, -1]
        self.UI.arrow_angle     = [-1]
        self.UI.Result_path     = None
        self.UI.Coord_path      = None
        self.UI.showResult_path = None
        self.UI.draw_pix_radius = self.UI.spinBox_radius.value()
        self.UI.draw_pix_cursor = 1
        self.UI.tomo_result_select  = []
        self.UI.tomo_result_select_all = {1:[]}
        #self.UI.tomo_result_back    = []
        #self.UI.tomo_result_show    = []
        #self.UI.tomo_result_stack   = []
        self.UI.surf_xyz            = []
        self.UI.surf_mode           = []
        self.UI.surf_direction      = []
        self.UI.show_mrc_flag       = False
        self.UI.tomo_check_current  = 0
        self.UI.tomo_result_current = 0
        self.UI.CoordMapping        = None
        # self.UI.EraseFlag           = False
        self.UI.savetxt_path        = None
        self.UI.tomo_extract        = None
        self.UI.namesuffix          = None
        self.UI.Boundary_input_path = None
        self.UI.draw_pix_linewidth  = 1
        # self.UI.label_Press_Flag    = False
        self.UI.draw_changed_Flag   = False
        self.UI.select_group        = None
        self.UI.select_group_Flag   = False
        #self.UI.show_mrc_flag       = True
        self.UI.surf_right_pos      = None
        # reset all the AD setting
        self.UI.ad_minsurf          = 10
        self.UI.ad_ncpu             = 1
        self.UI.ad_expandratio      = 0
        self.UI.advance_setting.checkBox_adprinttime.setChecked(False)
        self.UI.label_zlimit.setText("None")
        self.UI.label_resultzlimit.setText("None")
        self.UI.spinBox_z.setValue(0)
        self.UI.spinBox_resultz.setValue(0)
        self.UI.graphicsScene.clear()
        self.UI.graphicsScene_resultside.clear()
        self.UI.graphicsScene_result.clear()
        self.UI.graphicsScene       = QGraphicsScene()
        self.UI.graphicsScene_result = QGraphicsScene()
        self.UI.graphicsScene_resultside = QGraphicsScene()
        for del_i in reversed(range(self.UI.scroll_hbox.count())):
            if self.UI.scroll_hbox.itemAt(del_i) is not None:
                self.UI.scroll_hbox.itemAt(del_i).layout().itemAt(0).widget().deleteLater()
                self.UI.scroll_hbox.itemAt(del_i).layout().itemAt(1).widget().deleteLater()
                self.UI.scroll_hbox.itemAt(del_i).layout().itemAt(2).widget().deleteLater()
                self.UI.scroll_hbox.itemAt(del_i).deleteLater()
        for del_i in reversed(range(self.UI.graphicsView.scrollArea_Select_vbox.count())):
            self.UI.graphicsView.scrollArea_Select_vbox.itemAt(del_i).layout().itemAt(1).widget().setParent(None)
            self.UI.graphicsView.scrollArea_Select_vbox.itemAt(del_i).layout().itemAt(0).widget().setParent(None)
            self.UI.graphicsView.scrollArea_Select_vbox.itemAt(del_i).layout().setParent(None)
        for del_i in reversed(range(self.UI.graphicsView.scrollArea_auto_Select_vbox.count())):
            self.UI.graphicsView.scrollArea_auto_Select_vbox.itemAt(del_i).widget().setParent(None)
        for del_i in reversed(range(self.UI.select_result_scroll_vbox.count())):
            self.UI.select_result_scroll_vbox.itemAt(del_i).widget().setParent(None)
        for del_i in reversed(range(self.UI.scrollArea_manualsurf_vbox.count())):
            self.UI.scrollArea_manualsurf_vbox.itemAt(del_i).layout().itemAt(1).widget().setParent(None)
            self.UI.scrollArea_manualsurf_vbox.itemAt(del_i).layout().itemAt(0).widget().setParent(None)
            self.UI.scrollArea_manualsurf_vbox.itemAt(del_i).layout().setParent(None)
        for del_i in reversed(range(self.UI.frame_manual.scrollArea_manualpoints_vbox.count())):
            self.UI.frame_manual.scrollArea_manualpoints_vbox.itemAt(del_i).widget().setParent(None)

        self.UI.Text_Information.clear()
        self.UI.Text_SavePath.clear()
        #pass


    def disabled_all_button(self):
        self.UI.Button_Openmask.setEnabled(False)
        self.UI.Button_Openraw.setEnabled(False)
        self.UI.Button_Switchshow.setEnabled(False)
        self.UI.Button_OpenSession.setEnabled(False)
        self.UI.Button_Selectsavepath.setEnabled(False)
        #self.UI.Button_Memseg.setEnabled(False)
        self.UI.Button_FindSurface.setEnabled(False)
        self.UI.Button_ExtractSurface.setEnabled(False)
        self.UI.label_progress.setText("Loading Session...Please Wait")

    def Reset_config_Button(self):
        if self.UI.openmemseg:
            self.UI.Button_Openmask.setEnabled(False)
        else:
            self.UI.Button_Openmask.setEnabled(True)
        self.UI.Button_Openraw.setEnabled(True)
        self.UI.Button_Switchshow.setEnabled(True)
        self.UI.Button_OpenSession.setEnabled(True)
        self.UI.Button_Selectsavepath.setEnabled(True)
        #self.UI.Button_Memseg.setEnabled(True)
        self.UI.Button_FindSurface.setEnabled(True)
        self.UI.Button_ExtractSurface.setEnabled(True)
        if len(self.UI.tomo_select_surf) != 0:
            self.UI.tabWidget.setCurrentIndex(0)
            self.UI.tabBarInit(0)
        self.UI.label_progress.setText("Load Session Done")

    def config_set_raw_enabled(self,bool):
        self.actionRaw.setEnabled(bool)
        self.UI.Init_MrcImage()

    def config_set_mask_enabled(self,bool):
        self.actionMask.setEnabled(bool)

    def config_set_boundary_enabled(self,bool):
        self.actionBoundary.setEnabled(bool)

    def config_set_save_path_text(self,string):
        self.UI.Text_SavePath.setText(string)
        self.UI.pickfolder = os.path.dirname(string)

    def config_set_select_points(self,number):
        select_label        = QLabel()
        select_label.setContextMenuPolicy(Qt.CustomContextMenu)
        select_label.customContextMenuRequested.connect(self.Qlabel_Menu)
        select_label.setText(
            f"{number}. Surface")
        select_check        = QCheckBox()
        select_hbox         = QHBoxLayout()
        select_hbox.addWidget(select_label)
        select_hbox.addWidget(select_check)
        self.UI.graphicsView.scrollArea_Select_vbox.addLayout(select_hbox)
        #self.UI.graphicsView.scrollArea_Select_vbox.addWidget(select_label)
        self.UI.graphicsView.scrollArea_Select_widget.setLayout(self.UI.graphicsView.scrollArea_Select_vbox)
        self.UI.graphicsView.scrollArea_Select.setWidget(self.UI.graphicsView.scrollArea_Select_widget)

    def Qlabel_Menu(self):
        self.Surf_popMenu = QMenu()
        for i in range(self.UI.graphicsView.scrollArea_Select_vbox.count()):
            chosed_label        = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()
            x, y                = self.UI.surf_right_pos.x(),self.UI.surf_right_pos.y()
            pic_xmin, pic_ymin  = chosed_label.x(), chosed_label.y()
            pic_xmax, pic_ymax = pic_xmin + chosed_label.width(), pic_ymin + chosed_label.height()
            if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                self.delete_surf_i      = i
                self.delete_surf        = chosed_label
                self.delete_action      = QAction(f'Delete {chosed_label.text()}', self)
                self.show_action        = QAction(f'Show {chosed_label.text()}',self)
                self.UI.surf_right_pos  = None
                break
        if isinstance(self.delete_action,QAction):
            self.delete_action.triggered.connect(self.Delete_surf)
            self.show_action.triggered.connect(self.Show_surf)
            self.Surf_popMenu.addAction(self.show_action)
            self.Surf_popMenu.addAction(self.delete_action)
            self.delete_select_action = QAction(f'Delete Selected', self)
            self.show_select_action = QAction(f'Show Selected', self)
            self.show_all_action = QAction(f'Show All', self)
            self.close_all_action = QAction(f'Hide All', self)
            if isinstance(self.delete_select_action, QAction):
                self.delete_select_action.triggered.connect(self.UI.graphicsView.Delete_select_surf)
                self.show_select_action.triggered.connect(self.UI.graphicsView.Show_selected_surf)
                self.show_all_action.triggered.connect(self.UI.graphicsView.Show_all_action)
                self.close_all_action.triggered.connect(self.UI.graphicsView.Close_all_surf)
                self.Surf_popMenu.addAction(self.show_select_action)
                self.Surf_popMenu.addAction(self.show_all_action)
                self.Surf_popMenu.addAction(self.close_all_action)
                self.Surf_popMenu.addAction(self.delete_select_action)
            self.Surf_popMenu.move(QCursor.pos())
            self.Surf_popMenu.show()

    def Show_surf(self):
        Checkbox = self.UI.graphicsView.scrollArea_Select_vbox.itemAt(self.delete_surf_i).layout().itemAt(1).widget()
        Checkbox.setChecked(True)
        self.UI.graphicsView.Show_selected_surf()

    def Delete_surf(self):
        if self.UI.pixmap is not None:
            if self.UI.show_ask_surf is False:
                self.Remove_surf_widget()
            else:
                self.scroll_remove_popupInfo()


    def scroll_remove_popupInfo(self):
        msg = QMessageBox()
        msg.setWindowTitle("Delete Warning")
        num = self.delete_surf.text().split(".")[0]
        msg.setText(f'The Select Result {self.delete_surf.text()} will be deteled!\n'
                    f'And the relevant Extract Result (like {num}-1,{num}-2...) will be deleted too! '
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
            self.Remove_surf_widget()
            if cb.isChecked():
                self.UI.show_ask_surf = False
        else:
            if cb.isChecked():
                self.UI.show_ask_surf = False

    def Remove_surf_widget(self):
            del self.UI.tomo_select_surf[self.delete_surf_i]
            del self.UI.tomo_select_surf_xyz[self.delete_surf_i]
            del self.UI.tomo_select_surf_mode[self.delete_surf_i]
            del self.UI.tomo_select_surf_direction[self.delete_surf_i]
            #del self.UI.minsurf[self.delete_surf_i]
            self.UI.graphicsView.scrollArea_Select_vbox.itemAt(self.delete_surf_i).layout().itemAt(1).widget().setParent(None)
            self.UI.graphicsView.scrollArea_Select_vbox.itemAt(self.delete_surf_i).layout().itemAt(0).widget().setParent(None)
            self.UI.graphicsView.scrollArea_Select_vbox.itemAt(self.delete_surf_i).layout().setParent(None)
            if self.delete_surf_i < self.UI.tomo_surfcurrent:
                self.UI.tomo_surfcurrent    = len(self.UI.tomo_select_surf) - 1
                self.UI.tomo_check_current  = self.UI.tomo_check_current - 1
            elif self.delete_surf_i > self.UI.tomo_surfcurrent:
                pass
            else:
                self.UI.tomo_surfcurrent    = len(self.UI.tomo_select_surf) - 1
                self.UI.tomo_check_current  = 0
                self.UI.graphicsView.ClearReference()
                #self.UI.graphicsView.Reset_scrollArea_Select()
                self.UI.SURF_path = None
                self.UI.tomo_extract = None
                self.UI.showMrcImage()

            num = self.delete_surf.text().split(".")[0]
            for del_i in reversed(range(0,self.UI.scroll_hbox.count())):
                chosed_label = self.UI.scroll_hbox.itemAt(del_i).layout()
                self.UI.select_pic = chosed_label.itemAt(1).widget()
                self.UI.select_label = chosed_label.itemAt(0).widget()
                self.UI.select_path = chosed_label.itemAt(2).widget()
                self.UI.select_layout = self.UI.scroll_hbox.itemAt(del_i)
                if self.UI.select_label.text().split("-")[0] == num:
                    #print("it will delete ",self.UI.select_label.text())
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
            self.UI.graphicsView.Show_selected_surf()
            delete_path = os.path.join(self.UI.Text_SavePath.toPlainText(), "surface_" + num + "_" +
                               os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
            if os.path.exists(delete_path):
                #print("delete ",delete_path)
                self.remove_surface(delete_path)

    def config_set_scroll_surf(self,bool):
        self.UI.graphicsView.Reset_scrollArea_Select()
        self.UI.graphicsView.Reset_scrollArea_Select_points()
        self.UI.Picked_point = self.UI.tomo_select[self.UI.tomo_current]
        if bool:
            self.UI.waitCondition.wakeAll()
        # 2022/7/20 self.UI.graphicsView.Reset_surf_show_parameter()
        #self.UI.Init_MrcImage()

    def config_set_select_manual(self,bool):
        select_label = QLabel()
        select_label.setContextMenuPolicy(Qt.CustomContextMenu)
        select_label.customContextMenuRequested.connect(self.Qlabel_Menu_manual)
        select_label.setText(f"{self.UI.tomo_manual_number}. Surface")
        select_check = QCheckBox()
        select_hbox = QHBoxLayout()
        select_hbox.addWidget(select_label)
        select_hbox.addWidget(select_check)
        self.UI.frame_manual.scrollArea_manualsurf_vbox.addLayout(select_hbox)
        #self.UI.scrollArea_manualsurf_vbox.addWidget(select_label)
        self.UI.frame_manual.scrollArea_manualsurf_widget.setLayout(self.UI.frame_manual.scrollArea_manualsurf_vbox)
        self.UI.frame_manual.scrollArea_manualsurf.setWidget(
            self.UI.frame_manual.scrollArea_manualsurf_widget)
        if bool:
            self.UI.waitCondition.wakeAll()

    def Qlabel_Menu_manual(self):
        self.Surf_popMenu_manual = QMenu()
        x, y = self.UI.surf_right_pos.x(), self.UI.surf_right_pos.y()
        for i in reversed(range(self.UI.frame_manual.scrollArea_manualsurf_vbox.count())):
            chosed_label        = self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(0).widget()
            pic_xmin, pic_ymin  = chosed_label.x(), chosed_label.y()
            pic_xmax, pic_ymax  = pic_xmin + chosed_label.width(), pic_ymin + chosed_label.height()
            if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                self.delete_manual_i            = i
                self.delete_manual              = chosed_label
                self.delete_action_manual       = QAction(f'Delete "{chosed_label.text()}"', self)
                self.show_action_manual         = QAction(f'Show {chosed_label.text()}',self)
                self.UI.surf_right_pos          = None
                break
        if isinstance(self.delete_action_manual,QAction):
            self.delete_action_manual.triggered.connect(self.Delete_manual)
            self.Surf_popMenu_manual.addAction(self.delete_action_manual)
            self.show_action_manual.triggered.connect(self.Show_manual)
            self.Surf_popMenu_manual.addAction(self.show_action_manual)
            # Rest of func
            self.manual_delete_select_action = QAction(f'Delete Selected', self)
            self.manual_show_select_action = QAction(f'Show Selected', self)
            self.manual_show_all_action = QAction(f'Show All', self)
            self.manual_close_all_action = QAction(f'Hide All', self)
            self.manual_delete_select_action.triggered.connect(self.UI.frame_manual.Delete_select_manual)
            self.manual_show_select_action.triggered.connect(self.UI.frame_manual.Show_selected_manual)
            self.manual_show_all_action.triggered.connect(self.UI.frame_manual.Show_all_manual)
            self.manual_close_all_action.triggered.connect(self.UI.frame_manual.Close_all_manual)
            self.Surf_popMenu_manual.addAction(self.manual_show_select_action)
            self.Surf_popMenu_manual.addAction(self.manual_show_all_action)
            self.Surf_popMenu_manual.addAction(self.manual_close_all_action)
            self.Surf_popMenu_manual.addAction(self.manual_delete_select_action)
            self.Surf_popMenu_manual.move(QCursor.pos())
            self.Surf_popMenu_manual.show()

    def Show_manual(self):
        Checkbox = self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(
            self.delete_manual_i).layout().itemAt(1).widget()
        Checkbox.setChecked(True)
        self.UI.frame_manual.Show_selected_manual()

    def Delete_manual(self):
        if self.UI.pixmap is not None:
            if self.UI.show_ask_surf is False:
                self.Remove_manual_widget()
            else:
                self.scroll_remove_popupInfo_manual()


    def scroll_remove_popupInfo_manual(self):
        msg = QMessageBox()
        msg.setWindowTitle("Delete Warning")
        num = self.delete_manual.text().split(".")[0]
        msg.setText(f'The Select Manual Result {self.delete_manual.text()} will be deteled!\n'
                    f'And the relevant Extract Result (like M{num}-1,M{num}-2...) will be deleted too! '
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
                self.UI.show_ask_surf = False
        else:
            if cb.isChecked():
                self.UI.show_ask_surf = False

    def remove_surface(self,dir):
        self.UI.tomo_addresult = None
        if(os.path.isdir(dir)):
            for file in os.listdir(dir):
                self.remove_surface(os.path.join(dir,file))
            if (os.path.exists(dir)):
                os.rmdir(dir)
        else:
            if(os.path.exists(dir)):
                os.remove(dir)

    def Remove_manual_widget(self):
        del self.UI.tomo_manual_surf[self.delete_manual_i]
        self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(self.delete_manual_i).layout().itemAt(1).widget().setParent(None)
        self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(self.delete_manual_i).layout().itemAt(0).widget().setParent(None)
        self.UI.frame_manual.scrollArea_manualsurf_vbox.itemAt(self.delete_manual_i).layout().setParent(None)
        if self.delete_manual_i < self.UI.tomo_manual_surfcurrent:
            self.UI.tomo_manual_surfcurrent = len(self.UI.tomo_manual_surf) - 1
            self.UI.tomo_check_current      = self.UI.tomo_check_current - 1
        elif self.delete_manual_i > self.UI.tomo_manual_surfcurrent:
            pass
        else:
            self.UI.tomo_manual_surfcurrent = len(self.UI.tomo_manual_surf) - 1
            self.UI.tomo_check_current      = 0
            self.UI.tomo_manual_current     = 0
            self.UI.tomo_manual_select      = []
            self.UI.frame_manual.ClearReference()
            self.UI.showMrcImage()
            self.UI.frame_manual.Reset_scrollArea_Select()
        num = self.delete_manual.text().split(".")[0]
        for del_i in reversed(range(0,self.UI.scroll_hbox.count())):
            chosed_label = self.UI.scroll_hbox.itemAt(del_i).layout()
            self.UI.select_pic = chosed_label.itemAt(1).widget()
            self.UI.select_label = chosed_label.itemAt(0).widget()
            self.UI.select_path = chosed_label.itemAt(2).widget()
            self.UI.select_layout = self.UI.scroll_hbox.itemAt(del_i)
            if self.UI.select_label.text().split("-")[0][1:] == num:
                #print("it will delete ",self.UI.select_label.text())
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

        self.UI.frame_manual.Show_selected_manual()
        delete_path = os.path.join(self.UI.Text_SavePath.toPlainText(), "manual_" + num + "_" +
                                os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
        if os.path.exists(delete_path):
            #print("delete ", delete_path)
            self.remove_surface(delete_path)


    def config_set_select_manual_points(self,bool):
        self.UI.frame_manual.Reset_scrollArea_Surf_Select()
        self.UI.frame_manual.Reset_scrollArea_Select()
        self.UI.frame_manual.scrollArea_manualpoints_widget.setLayout(
            self.UI.frame_manual.scrollArea_manualpoints_vbox)
        self.UI.frame_manual.scrollArea_manualpoints.setWidget(
            self.UI.frame_manual.scrollArea_manualpoints_widget)
        #self.UI.Init_MrcImage()
        pass

    def config_set_extract_surf(self,file):
        chose_index = file.split("_")[1]
        each_vbox = QVBoxLayout()
        Name_label = QLabel()
        if file.split("_")[0] == "manual":
            chose_index = "M" + chose_index
        Name_label.setText(chose_index)
        Name_label.setScaledContents(True)
        Font = QFont("Agency FB", 10)
        Font.setBold(True)
        Name_label.setFont(Font)
        Path_label = QLabel()
        Path_label.setText(self.UI.Result_path)
        Path_label.setHidden(True)
        with mrcfile.mmap(self.UI.Result_path) as mrc:
            self.UI.tomo_addresult = mrc.data[:, ::-1, :]
            # tomo_min = np.min(self.UI.tomo_addresult)
            # tomo_max = np.max(self.UI.tomo_addresult)
            # self.UI.tomo_addresult = (
            #         (self.UI.tomo_addresult - tomo_min) / (tomo_max - tomo_min) * 255).astype(
            #     np.uint8)
        # ==== config extract ====
        Pro_label = QLabel()
        mrc_image_Image = self.UI.tomo_addresult[int(self.UI.tomo_addresult.shape[0] / 2), :, :]
        tomo_min = np.min(mrc_image_Image).astype(float)
        tomo_max = np.max(mrc_image_Image).astype(float)
        mrc_image_Image = (mrc_image_Image - tomo_min) / (tomo_max - tomo_min) * 255
        mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
        mrc_image_Image = np.asarray(mrc_image_Image)
        height, width, channels = mrc_image_Image.shape
        bytesPerLine = channels * width
        qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = QPixmap.fromImage(qImg)
        Pro_label.setPixmap(qImg)
        Pro_label.setScaledContents(True)
        Pro_label.setFixedSize(int(220 * width / height), 220)
        each_vbox.addWidget(Name_label)
        each_vbox.addWidget(Pro_label)
        each_vbox.addWidget(Path_label)
        each_vbox.setSpacing(self.UI.scrollVspacing)
        Pro_label.setToolTip(self.UI.Result_path)
        QToolTip.setFont(QFont('Agency FB', 10))
        self.UI.scroll_hbox.addLayout(each_vbox)
        self.UI.scroll_hbox.setSpacing(self.UI.scrollHspacing)
        self.UI.scroll_widget.setLayout(self.UI.scroll_hbox)
        self.UI.scrollArea.setWidget(self.UI.scroll_widget)
        self.UI.waitCondition.wakeAll()

    def config_reset_extract_surf(self,file):
        pass

    def config_load_progress(self, val):
        self.UI.progressBar.setValue(val)

    def clicker_Getmask(self):
        if self.UI.Text_SavePath.toPlainText() == "" or self.UI.MRC_path is None:
            QMessageBox.warning(self, "Get Mask Warning", 
                                "you need provide Raw tomo and output path at first", 
                                QMessageBox.Ok)
            return
        try:
            import torch
            # from membrane_seg import segmentation
            # from seg_postprocess import post_process
            # self.UI.Button_Openmask.setEnabled(False)
            self.UI.memseg = UI_Mpicker_FrameMemseg()
            self.UI.memseg.setParameters(self.UI)
            self.UI.memseg.show()
            # self.UI.openmemseg = True
        except Exception as e:
            # self.UI.Button_Openmask.setEnabled(True)
            str_e = repr(e)
            QMessageBox.warning(self, "Get Mask Warning", str_e, QMessageBox.Ok)

    def clicker_Openmask(self):
        if self.UI.Text_SavePath.toPlainText() == "" or self.UI.MRC_path is None:
            QMessageBox.warning(self, "Get Mask Warning", 
                                "you need provide Raw tomo and output path at first", 
                                QMessageBox.Ok)
            return
        if self.UI.MRC_path is not None:
            startpath = os.path.join(self.UI.Text_SavePath.toPlainText(), "memseg")
            if not os.path.isdir(startpath):
                startpath = os.getcwd()
            fname = QFileDialog.getOpenFileName(self,"Open Mask File",startpath,
                                                "All Files (*);;MRC Files (*.mrc);;Jpg File (*,jpg)")
            if fname[0] != '':
                try:
                    self.UI.MASK_path = fname[0]
                    self.disabled_all_button()
                    self.UI.show_mrc_flag = False
                    self.Thread_OpenMask = QThread_OpenMask()
                    self.Thread_OpenMask.setParameters(self.UI)
                    self.Thread_OpenMask.error.connect(self.openMaskError)
                    self.Thread_OpenMask.shapeerror.connect(self.openMaskShapeError)
                    self.Thread_OpenMask.finished.connect(self.finished_Open_mask)
                    #self.Thread_OpenMask.finished.connect(self.finished_Open_file)
                    self.Thread_OpenMask.start()
                    #self.actionGet.setEnabled(False)
                except:
                    self.openMaskError()
        else:
            QMessageBox.warning(self, "Input Warning",
                                "Please select RAW file before open a mask. ", QMessageBox.Ok)

    def openMaskShapeError(self):
        self.Thread_OpenMask.finished.disconnect()
        self.Thread_OpenMask.terminate()
        self.UI.tomo_show = self.UI.tomo_data
        self.UI.show_mrc_flag = True
        self.UI.showMrcImage()
        self.Reset_config_Button()
        self.actionMask.setEnabled(True)
        self.actionBoundary.setEnabled(True)
        QMessageBox.warning(self, "Input Warning",
                            "Input Mask shape is different from Raw shape. "
                            "Please check your input Mask. ", QMessageBox.Ok)
    def openMaskError(self):
        self.Thread_OpenMask.finished.disconnect()
        self.Thread_OpenMask.terminate()
        self.UI.tomo_show = self.UI.tomo_data
        self.UI.show_mrc_flag = True
        self.UI.showMrcImage()
        self.Reset_config_Button()
        self.actionMask.setEnabled(True)
        self.actionBoundary.setEnabled(True)
        str_e = repr(self.Thread_OpenMask.exception)
        QMessageBox.warning(self, "Input Warning",
                            str_e+"\nPlease check your input file. ", QMessageBox.Ok)

    def finishedopenError(self):
        self.Thread_FinishedOpen.finished.disconnect()
        self.Thread_FinishedOpen.terminate()
        self.Reset_config_Button()
        self.actionMask.setEnabled(True)
        self.actionBoundary.setEnabled(True)
        str_e = repr(self.Thread_OpenMask.exception)
        QMessageBox.warning(self, "Open Error",
                            + str_e + "\nPlease check your input file! ", QMessageBox.Ok)

    def finished_Open_mask(self):
        if self.UI.tomo_mask is not None:
            self.Reset_config_Button()
            self.actionMask.setEnabled(True)
            self.actionBoundary.setEnabled(True)
            #self.UI.Init_MrcImage()
            self.finished_Open()
            self.UI.showMrcImage()
            # if self.UI.Text_SavePath.toPlainText() != "" and self.UI.MRC_path is not None:
            #     config_name = os.path.splitext(os.path.basename(self.UI.MRC_path))[0]
            #     # print("self.UI.pickfolder = ", self.UI.pickfolder)
            #     # print("self.UI.Text_SavePath = ", self.UI.Text_SavePath.toPlainText())
            #     if self.UI.Text_SavePath.toPlainText() == self.UI.pickfolder:
            #         if os.path.exists(os.path.join(self.UI.pickfolder, config_name)) is False:
            #             os.mkdir(os.path.join(self.UI.pickfolder, config_name))
            #             self.UI.ini_config_path = os.path.join(
            #                 os.path.abspath(self.UI.pickfolder),
            #                 config_name + ".config")
            #             # print("New iniconfig = ",self.UI.ini_config_path)
            #         else:
            #             # config_name = os.path.splitext(config_name)
            #             # os.mkdir(os.path.join(self.UI.pickfolder, config_name))
            #             self.UI.ini_config_path = os.path.join(
            #                 os.path.abspath(self.UI.pickfolder),
            #                 config_name + ".config")
            #         self.UI.Text_SavePath.setText(os.path.join(self.UI.pickfolder, config_name))
            #     else:
            #         self.UI.ini_config_path = os.path.join(
            #             os.path.abspath(self.UI.pickfolder),
            #             config_name + ".config")
            #self.UI.Init_config_all()

    # def finished_Open_file(self):
    #     if self.UI.tomo_mask is not None:
    #         self.UI.Init_MrcImage()
    #         #self.Thread_Openconfig.start()
    #         #self.Thread_FinishedOpen.start()
    #         self.finished_Open()

    def finished_Open(self, savepath_click=False):
        if self.UI.Text_SavePath.toPlainText() != "" and self.UI.MRC_path is not None:
            config_name = os.path.splitext(os.path.basename(self.UI.MRC_path))[0]
            # print("self.UI.pickfolder = ", self.UI.pickfolder)
            # print("self.UI.Text_SavePath = ", self.UI.Text_SavePath.toPlainText())
            # if self.UI.Text_SavePath.toPlainText() == self.UI.pickfolder:
            if os.path.exists(os.path.join(self.UI.pickfolder, config_name)) is False:
                os.mkdir(os.path.join(self.UI.pickfolder, config_name))
            self.UI.ini_config_path = os.path.join(
                os.path.abspath(self.UI.pickfolder),
                config_name + ".config")
                #print("New iniconfig = ",self.UI.ini_config_path)
            # else:
            #     #config_name = os.path.splitext(config_name)
            #     #os.mkdir(os.path.join(self.UI.pickfolder, config_name))
            #     self.UI.ini_config_path = os.path.join(
            #         os.path.abspath(self.UI.pickfolder),
            #         config_name + ".config")
            class_file_dir = os.path.join(self.UI.pickfolder, config_name)
            self.UI.setup_class_file(class_file_dir)
            self.UI.Text_SavePath.setText(os.path.join(self.UI.pickfolder, config_name))
            # else:
            #     self.UI.ini_config_path = os.path.join(
            #         os.path.abspath(self.UI.pickfolder),
            #         config_name + ".config")
            #print("self.iniconfig = ",self.UI.ini_config_path)
            #print("exits = ",os.path.exists(self.UI.ini_config_path))
            #print("self.UI.ini_config_path = ",self.UI.ini_config_path)
            if os.path.exists(self.UI.ini_config_path) and savepath_click:
                # only when choose save path. not when open raw or mask.
                # copy from Open_config()
                try:
                    self.clear_all_widgets()
                    self.UI.ini_config.read(self.UI.ini_config_path, encoding='utf-8')
                    self.Thread_Openconfig.setParameters(self.UI)
                    self.Thread_Openconfig.start()
                    self.disabled_all_button()
                except Exception as e:
                    str_e = repr(e)
                    QMessageBox.warning(self,"Input Warning",str_e,QMessageBox.Ok)
            else:
                self.UI.Init_config_all()

    def clicker_openRaw(self):
        fname = QFileDialog.getOpenFileName(self,"Open Raw File",os.getcwd(),
                                            "All Files (*);;PNG Files (*.png);;Jpg File (*,jpg)")
        if fname[0] != '':
            try:
                with mrcfile.mmap(fname[0]) as mrc:
                    tomo_data = mrc.data[:,::-1,:] # .astype(np.float32)
                    # tomo_min = np.min(tomo_data)
                    # tomo_max = np.max(tomo_data)
                    # self.UI.tomo_data = ((tomo_data - tomo_min)/(tomo_max - tomo_min) *255).astype(np.uint8)
                if self.UI.MRC_path is not None:
                    self.clear_all_widgets()
                self.UI.tomo_show = tomo_data
                self.UI.tomo_data = tomo_data
                self.UI.show_mrc_flag = True
                self.UI.Init_MrcImage()
                self.UI.MRC_path = fname[0]
                self.actionRaw.setEnabled(True)
                #self.finished_Open_file()
                self.finished_Open()
                self.actionMask.setDisabled(True)
                self.actionBoundary.setDisabled(True)
            except Exception as e:
                str_e = repr(e)
                QMessageBox.warning(self,"Input Warning",str_e,QMessageBox.Ok)

    def clicker_savePath(self):
        if self.UI.MRC_path is not None:
            fname = QFileDialog.getExistingDirectory(self, "Open Save Folder",
                                                 os.getcwd())
            if fname != '':
                try:
                    temp_mrc_path       = self.UI.MRC_path
                    temp_mask_path      = self.UI.MASK_path
                    temp_boundary_path  = self.UI.Boundary_path
                    temp_mrc_data       = self.UI.tomo_data
                    temp_mask_data      = self.UI.tomo_mask
                    temp_boundary_data  = self.UI.tomo_boundary
                    temp_show_data      = self.UI.tomo_show
                    if self.UI.Text_SavePath.toPlainText() != "":
                        self.clear_all_widgets()
                        self.UI.MRC_path        = temp_mrc_path
                        self.UI.MASK_path       = temp_mask_path
                        self.UI.Boundary_path   = temp_boundary_path
                        self.UI.tomo_data       = temp_mrc_data
                        self.UI.tomo_mask       = temp_mask_data
                        self.UI.tomo_boundary   = temp_boundary_data
                        self.UI.tomo_show       = temp_mrc_data
                    self.UI.Text_SavePath.setText(fname)
                    self.UI.pickfolder = fname
                    self.finished_Open(savepath_click=True)
                except Exception as e:
                    str_e = repr(e)
                    QMessageBox.warning(self, "Input Warning", str_e, QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "Input Warning",
                                "Please select RAW file before open a mask. ", QMessageBox.Ok)

    def clicker_HideSurface(self):
        if self.UI.pixmap is not None:
            #self.UI.Load_config_surface()
            if self.UI.Hide_Surf_Flag == False:
                self.UI.Hide_Surf_Flag = True
                self.UI.Button_Hidesurf.setStyleSheet("background-color: gray")
            else:
                self.UI.Hide_Surf_Flag = False
                self.UI.Button_Hidesurf.setStyleSheet("background-color: None")
            self.UI.showMrcImage()

    def slide_Contrast(self, value):
        if self.UI.pixmap is not None:
            self.UI.slider_Contrast_value = round(value / 10,1)
            self.UI.mrc_tomo_min = self.UI.tomo_show_mean + (self.UI.slider_Bright_value -
                                                       self.UI.slider_Contrast_value) * self.UI.tomo_show_std
            self.UI.mrc_tomo_max = self.UI.tomo_show_mean + (self.UI.slider_Bright_value +
                                                       self.UI.slider_Contrast_value) * self.UI.tomo_show_std
            if self.UI.mrc_tomo_max != self.UI.mrc_tomo_min:
                self.UI.mrc_contrast = 255 / (
                            self.UI.mrc_tomo_max - self.UI.mrc_tomo_min)  # / (self.UI.old_mrc_tomo_max - self.UI.old_mrc_tomo_min)
                self.UI.showMrcImage()
            else:
                self.UI.mrc_contrast = 255
                self.UI.showMrcImage()

    def slide_Bright(self, value):
        if self.UI.pixmap is not None:
            self.UI.slider_Bright_value = round(value / 10,1)
            self.UI.mrc_tomo_min = self.UI.tomo_show_mean + (self.UI.slider_Bright_value -
                                                             self.UI.slider_Contrast_value) * self.UI.tomo_show_std
            self.UI.mrc_tomo_max = self.UI.tomo_show_mean + (self.UI.slider_Bright_value +
                                                             self.UI.slider_Contrast_value) * self.UI.tomo_show_std
            if self.UI.mrc_tomo_max != self.UI.mrc_tomo_min:
                self.UI.mrc_contrast = 255 / (
                            self.UI.mrc_tomo_max - self.UI.mrc_tomo_min)  # / (self.UI.old_mrc_tomo_max - self.UI.old_mrc_tomo_min)
                self.UI.showMrcImage()
            else:
                self.UI.mrc_contrast = 255
                self.UI.showMrcImage()


    def doubleSpinBox_L_Contrast(self):
        if self.UI.pixmap is not None:
            self.UI.slider_Contrast_value = self.UI.doubleSpinBox_Contrast.value()
            self.UI.horizontalSlider_Contrast.setValue(self.UI.slider_Contrast_value * 10)
            self.UI.showMrcImage()
        else:
            pass

    def doubleSpinBox_L_Bright(self):
        if self.UI.pixmap is not None:
            self.UI.slider_Bright_value = self.UI.doubleSpinBox_Bright.value()
            self.UI.horizontalSlider_Bright.setValue(self.UI.slider_Bright_value * 10)
            self.UI.showMrcImage()
        else:
            pass

    def SpinBox_X(self):
        if self.UI.pixmap is not None:
            self.UI.realx = self.UI.doubleSpinBox_X.value()
            self.UI.showMrcImage()
            #self.UI.graphicsView.refresh_Cursor()

    def SpinBox_Y(self):
        if self.UI.pixmap is not None:
            self.UI.realy = self.UI.doubleSpinBox_Y.value()
            self.UI.showMrcImage()
            #self.UI.graphicsView.refresh_Cursor()

    def SpinBox_Z(self):
        if self.UI.pixmap is not None:
            self.UI.realz = int(self.UI.doubleSpinBox_Z.value())
            self.UI.showMrcImage()


    def slide_z(self,value):
        if self.UI.pixmap is not None:
            self.UI.realz = value
            # self.UI.mrc_image = self.UI.tomo_show[value-1, :, :]
            self.UI.showMrcImage()

    def SpinBox_z(self):
        if self.UI.pixmap is not None:
            value_z = self.UI.spinBox_z.value()
            if value_z >= 1 and value_z <= self.UI.tomo_show.shape[0]:
                self.UI.realz = value_z
            elif value_z < 1:
                self.UI.realz = 1
            else:
                self.UI.realz = self.UI.tomo_show.shape[0]
            self.UI.showMrcImage()

    def showRaw(self):
        if self.UI.tomo_show is not None:
            if self.UI.tomo_data is not None:
                self.UI.tomo_show = self.UI.tomo_data
                self.UI.show_mrc_flag = True
                self.UI.graphicsScene.clear()
                self.UI.showMrcImage()
            else:
                self.clicker_openRaw()

    def showMask(self):
        if self.UI.tomo_show is not None:
            #print("self.UI.tomo_mask = ",self.UI.tomo_mask)
            if self.UI.tomo_mask is not None:
                self.UI.tomo_show = self.UI.tomo_mask
                self.UI.show_mrc_flag = False
                self.UI.graphicsScene.clear()
                self.UI.showMrcImage()
            else:
                self.clicker_Openmask()

    def showBoundary(self):
        if self.UI.tomo_show is not None:
            if self.UI.tomo_boundary is not None:
                self.UI.tomo_show = self.UI.tomo_boundary
                self.UI.show_mrc_flag = False
                self.UI.graphicsScene.clear()
                self.UI.showMrcImage()
            else:
                self.clicker_openBoundary()

    def clicker_openBoundary(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "All Files (*);;Mrc Files (*.mrc)")
        if fname[0] != '':
            try:
                self.UI.Boundary_input_path = fname[0]
                with mrcfile.mmap(self.UI.Boundary_input_path) as mrc:
                    self.UI.tomo_boundary = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.UI.tomo_boundary)
                    # tomo_max = np.max(self.UI.tomo_boundary)
                    # if tomo_max != tomo_min:
                    #     self.UI.tomo_boundary = ((self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(
                    #         np.uint8)
                    # else:
                    #     self.UI.tomo_boundary = np.zeros(self.UI.tomo_boundary.shape)
                self.UI.tomo_show = self.UI.tomo_boundary
                self.UI.show_mrc_flag = False
                self.UI.graphicsScene.clear()
                self.UI.showMrcImage()
            except:
                QMessageBox.warning(self, "No Boundary Find",
                                    "Please Check path to Boundary",
                                    QMessageBox.Ok)

class QThread_OpenConfig(QThread):
    def __init__(self):
        super(QThread_OpenConfig,self).__init__()
        self.progress   = 0
        self.exception  = None
        self.exitcode   = False

    set_raw_enabled         = pyqtSignal(bool)
    set_mask_enabled        = pyqtSignal(bool)
    set_boundary_enabled    = pyqtSignal(bool)
    set_save_path_text      = pyqtSignal(str)
    set_select_points       = pyqtSignal(int)
    set_select_manual       = pyqtSignal(bool)
    set_select_manual_points= pyqtSignal(bool)
    set_scroll_surf         = pyqtSignal(bool)
    set_extract_surf        = pyqtSignal(str)
    load_progress           = pyqtSignal(int)

    def setParameters(self, UI):
        self.UI : Mpicker_gui.UI = UI

    def run(self):
        if self.UI.ini_config.has_section("Path"):
            config_name = os.path.splitext(os.path.basename(self.UI.ini_config_path))[0]
            self.set_save_path_text.emit(os.path.join(os.path.dirname(self.UI.ini_config_path), config_name))
            class_file_dir = os.path.join(os.path.dirname(self.UI.ini_config_path), config_name)
            self.UI.setup_class_file(class_file_dir)
            self.UI.show_mrc_flag = True
            if self.UI.ini_config.get('Path', 'InputRaw') != "None":
                self.UI.MRC_path = self.UI.ini_config.get('Path', 'InputRaw')
                with mrcfile.mmap(self.UI.MRC_path) as mrc:
                    self.UI.tomo_data = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.UI.tomo_data)
                    # tomo_max = np.max(self.UI.tomo_data)
                    # self.UI.tomo_data = ((self.UI.tomo_data - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    self.UI.tomo_show = self.UI.tomo_data
                    self.UI.show_mrc_flag = True
                self.set_raw_enabled.emit(True)
            else:
                self.UI.show_mrc_flag = False
                self.UI.MRC_path = None
            if self.UI.ini_config.get('Path', 'InputMask') != "None":
                self.UI.MASK_path = self.UI.ini_config.get('Path',"InputMask")
                with mrcfile.mmap(self.UI.MASK_path) as mrc:
                    self.UI.tomo_mask = mrc.data[:, ::-1, :]
                #     tomo_min = np.min(self.UI.tomo_mask)
                #     tomo_max = np.max(self.UI.tomo_mask)
                # if tomo_max != tomo_min:
                #     self.UI.tomo_mask = ((self.UI.tomo_mask - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                # else:
                #     self.UI.tomo_mask = (self.UI.tomo_mask * 255).astype(np.uint8)
                self.set_mask_enabled.emit(True)
            else:
                self.UI.MASK_path = None
            if self.UI.ini_config.get('Path', 'InputBoundary') != "None":
                self.UI.Boundary_path = self.UI.ini_config.get('Path', "InputBoundary")
                if os.path.exists(self.UI.Boundary_path):
                    with mrcfile.mmap(self.UI.Boundary_path) as mrc:
                        self.UI.tomo_boundary = mrc.data[:, ::-1, :]
                    #     tomo_min = np.min(self.UI.tomo_boundary)
                    #     tomo_max = np.max(self.UI.tomo_boundary)
                    # if tomo_max != tomo_min:
                    #     self.UI.tomo_boundary = ((self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    # else:
                    #     self.UI.tomo_boundary = (self.UI.tomo_boundary * 255).astype(np.uint8)
                    self.set_boundary_enabled.emit(True)
                else:
                    self.UI.ini_config.set('Path', "InputBoundary", "None")
            else:
                self.UI.Boundary_path = None
                if self.UI.MASK_path is not None:
                    self.UI.tomo_boundary = Get_Boundary(self.UI.MASK_path)
                    self.UI.Boundary_path = os.path.join(os.path.dirname(self.UI.ini_config_path), config_name,"my_boundary_6.mrc")
                    self.UI.Boundary_input_path = self.UI.Boundary_path
                    with mrcfile.new(self.UI.Boundary_path, overwrite=True) as mrc:
                        tomo_boundary = self.UI.tomo_boundary[:, ::-1, :]
                        mrc.set_data(tomo_boundary.astype(np.int8))
                    with mrcfile.mmap(self.UI.Boundary_path) as mrc:
                        self.UI.tomo_boundary = mrc.data[:, ::-1, :]
                    #self.UI.show_mrc_flag = False
                    self.set_boundary_enabled.emit(True)
                    self.UI.ini_config.set('Path', "InputBoundary",self.UI.Boundary_path)
                    with open(self.UI.ini_config_path, "w") as config_file:
                        self.UI.ini_config.write(config_file)
            self.show_progress(50)

            #==================== Load Mrc config ====================
            if self.UI.ini_config.get('Path',"Surface") != "None":
                Surface_string = self.UI.ini_config.get('Path', 'Surface')
                Surface_path   = os.path.join(os.path.dirname(self.UI.ini_config_path),config_name)
                Surface_list = Surface_string.split(" ")
                Surface_new_string = ""
                for Surface_str in Surface_list:
                    path = os.path.join(Surface_path, Surface_str)
                    if os.path.exists(path):
                        Surface_new_string = Surface_new_string + " " + Surface_str
                if Surface_new_string == "":
                    self.UI.ini_config.set('Path', 'Surface', "None")
                else:
                    self.UI.ini_config.set('Path', 'Surface', Surface_new_string)
            with open(self.UI.ini_config_path,"w") as config_file:
                self.UI.ini_config.write(config_file)
            #============== Load_cofig_surface =================
            Surface_string = self.UI.ini_config.get('Path', 'Surface')
            Surface_list = Surface_string.split(' ')
            Surface_auto_list = []
            Surface_manual_list = []
            if Surface_string != "None":
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
                        self.UI.result_folder = os.path.join(os.path.abspath(Surface_path),
                                                                 Surface_list_i)
                        self.UI.surf_config_path = os.path.join(self.UI.result_folder,
                                                          "surface_" + Surface_list_i.split("_")[1] + ".config")
                        self.UI.surf_config.read(self.UI.surf_config_path, encoding='utf-8')
                        Surface_number = self.UI.surf_config.getint("Parameter", "ID")
                        self.UI.tomo_select     = json.loads(self.UI.surf_config.get("Parameter", "Points"))
                        if self.UI.surf_config.get("Parameter", "Facexyz")[0] == "[":
                            self.UI.surf_mode       = json.loads(self.UI.surf_config.get("Parameter", "mode").replace('\'', '\"'))
                            self.UI.surf_xyz        = json.loads(self.UI.surf_config.get("Parameter", "Facexyz").replace('\'', '\"'))
                            self.UI.surf_direction  = json.loads(self.UI.surf_config.get("Parameter", "DirectionL2R").replace('\'', '\"'))
                        else:
                            self.UI.surf_mode       = [self.UI.surf_config.get("Parameter", "mode")]
                            self.UI.surf_xyz        = [self.UI.surf_config.get("Parameter", "Facexyz")]
                            self.UI.surf_direction  = [self.UI.surf_config.get("Parameter", "DirectionL2R")]
                            self.UI.surf_mode       *= len(self.UI.tomo_select)
                            self.UI.surf_xyz        *= len(self.UI.tomo_select)
                            self.UI.surf_direction  *= len(self.UI.tomo_select)
                            # print("Load from Surf file")
                            # print("self.UI.surf_mode = ", self.UI.surf_mode)
                            # print("self.UI.surf_xyz = ", self.UI.surf_xyz)
                            # print("self.UI.surf_direction = ", self.UI.surf_direction)
                        # if "minsurf" in self.UI.surf_config["Parameter"]:
                        #     self.UI.minsurf.append(self.UI.surf_config.getint("Parameter", "minsurf"))
                        # else:
                        #     self.UI.minsurf.append(10)
                        self.UI.tomo_select_surf.append(copy.deepcopy(self.UI.tomo_select))
                        self.UI.tomo_select_surf_xyz.append(copy.deepcopy(self.UI.surf_xyz))
                        self.UI.tomo_select_surf_mode.append(copy.deepcopy(self.UI.surf_mode))
                        self.UI.tomo_select_surf_direction.append(copy.deepcopy(self.UI.surf_direction))
                        self.UI.tomo_surfcurrent = len(self.UI.tomo_select_surf) - 1
                        self.UI.tomo_current = 0
                        self.set_select_points.emit(Surface_number)
                        self.UI.tomo_number = Surface_number + 1
                    self.UI.SURF_path = self.UI.surf_config_path.replace(".config", "_surf.mrc.npz")
                    # self.UI.tomo_extract = read_surface_mrc(self.UI.SURF_path, dtype=bool)[:, ::-1, :]
                    self.UI.tomo_extract = read_surface_coord(self.UI.SURF_path)
                    #self.UI.Picked_point = self.UI.tomo_select[self.UI.tomo_current]
                    self.set_scroll_surf.emit(True)
                    self.UI.mutex.lock()
                    self.UI.waitCondition.wait(self.UI.mutex)
                    self.UI.mutex.unlock()

                if len(Surface_manual_list) >0:
                    self.UI.tomo_manual_current = 0
                    for Surface_list_i in Surface_manual_list:
                        self.UI.result_folder = os.path.join(os.path.abspath(Surface_path),
                                                                 Surface_list_i)
                        self.UI.manual_config_path = os.path.join(self.UI.result_folder,
                                                                  Surface_list_i.split("_")[0] + "_"
                                                                  + Surface_list_i.split("_")[1] + ".config")
                        self.UI.manual_config.read(self.UI.manual_config_path, encoding='utf-8')
                        Surface_number      = self.UI.manual_config.getint("Parameter", "ID")
                        Surface_points      = self.UI.manual_config_path.replace(".config", "_surf.txt")
                        #Surface_points = self.UI.manual_config.get("Parameter", "txt_path")
                        self.UI.tomo_manual_select = np.loadtxt(Surface_points).astype(np.uint32)
                        if self.UI.tomo_manual_select.ndim == 1:
                            self.UI.tomo_manual_select = self.UI.tomo_manual_select.tolist()
                            self.UI.tomo_manual_select = [self.UI.tomo_manual_select]
                        else:
                            self.UI.tomo_manual_select = self.UI.tomo_manual_select.tolist()
                        self.UI.tomo_manual_surf.append(copy.deepcopy(self.UI.tomo_manual_select))
                        self.UI.tomo_manual_surfcurrent = len(self.UI.tomo_manual_surf) - 1
                        self.UI.ManualTxt_path = self.UI.manual_config_path.replace(".config", "_surf.txt")
                        self.UI.tomo_manual_number = Surface_number
                        self.set_select_manual.emit(True)
                        self.UI.mutex.lock()
                        self.UI.waitCondition.wait(self.UI.mutex)
                        self.UI.mutex.unlock()
                    self.UI.tomo_manual_number = Surface_number + 1
                    self.set_select_manual_points.emit(True)
                self.show_progress(75)
            #============== Load_config_extract ==================
                for surf_one in Surface_auto_list+Surface_manual_list:
                    surf_path       = os.path.join(Surface_path, surf_one)
                    all_file        = os.listdir(surf_path)
                    valid_file      = []

                    for file in all_file:
                        if "_result.mrc" in file:
                            configname = file.split("_")[0] + "_" + file.split("_")[1] + ".config"
                            if configname not in all_file:
                                continue
                            valid_file.append(file)
                    if len(valid_file) != 0:
                        for i in range(len(valid_file)):
                            extract_i   = int(valid_file[i].split("_")[1].split("-")[1])
                            extract_min = np.inf
                            extract_num = 0
                            for j in range(i+1,len(valid_file)):
                                extract_j = int(valid_file[j].split("_")[1].split("-")[1])
                                if extract_min > extract_j:
                                    extract_min = extract_j
                                    extract_num = j
                            if extract_i > extract_min:
                                temp_file               = valid_file[i]
                                valid_file[i]           = valid_file[extract_num]
                                valid_file[extract_num] = temp_file
                        for file in valid_file:
                            self.UI.Result_path = os.path.join(surf_path, file)
                            self.set_extract_surf.emit(file)
                            self.UI.mutex.lock()
                            self.UI.waitCondition.wait(self.UI.mutex)
                            self.UI.mutex.unlock()
            if self.UI.MRC_path is not None:
                self.UI.show_mrc_flag = True
            else:
                self.UI.show_mrc_flag = False
            self.show_progress(100)

    def show_progress(self,value):
        while self.progress < value:
            self.progress += 1
            self.load_progress.emit(self.progress)

    def kill_thread(self):
        self.terminate()


class QThread_OpenMask(QThread):
    def __init__(self):
        super(QThread_OpenMask,self).__init__()
        self.exception = None

    error         = pyqtSignal(bool)
    shapeerror    = pyqtSignal(bool)
    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        try:
            with mrcfile.mmap(self.UI.MASK_path) as mrc:
                tomo_mask = mrc.data[:, ::-1, :]
                # tomo_min = np.min(tomo_mask)
                # tomo_max = np.max(tomo_mask)
            if tomo_mask.shape == self.UI.tomo_data.shape:
                # self.UI.tomo_mask = ((tomo_mask - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                self.UI.tomo_mask = tomo_mask
                self.UI.tomo_show = self.UI.tomo_mask
                self.UI.tomo_boundary = Get_Boundary(self.UI.MASK_path) # array, not memmap
                # tomo_min = np.min(self.UI.tomo_boundary)
                # tomo_max = np.max(self.UI.tomo_boundary)
                # if tomo_max != tomo_min:
                #     self.UI.tomo_boundary = ((self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(
                #             np.uint8)
                # else:
                #     self.UI.tomo_boundary = np.zeros(self.UI.tomo_boundary.shape)
                # self.UI.openmemseg = True
                for i in range(1, self.UI.comboBox_Nearerode.count()):
                    boundarypath=os.path.join(self.UI.Text_SavePath.toPlainText(), 'my_boundary_' + self.UI.comboBox_Nearerode.itemText(i) + '.mrc')
                    if os.path.isfile(boundarypath):
                        try:
                            os.remove(boundarypath)
                            print("remove", boundarypath)
                        except Exception as e:
                            print(e)
                # self.UI.Init_config_all()
            else:
                self.shapeerror.emit(True)
        except Exception as e:
            self.exception = e
            self.error.emit(True)


class QThread_FinishedOpen(QThread):
    def __init__(self):
        super(QThread_FinishedOpen,self).__init__()
        self.exception = None

    error           = pyqtSignal(bool)
    set_config_all  = pyqtSignal(bool)
    set_surface     = pyqtSignal(bool)
    set_find        = pyqtSignal(bool)


    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        try:
            if self.UI.Text_SavePath.toPlainText() != "" and self.UI.MRC_path is not None:
                config_name = os.path.splitext(os.path.basename(self.UI.MRC_path))[0]
                # print("self.UI.pickfolder = ", self.UI.pickfolder)
                # print("self.UI.Text_SavePath = ", self.UI.Text_SavePath.toPlainText())
                if self.UI.Text_SavePath.toPlainText() == self.UI.pickfolder:
                    if os.path.exists(os.path.join(self.UI.pickfolder, config_name)) is False:
                        os.mkdir(os.path.join(self.UI.pickfolder, config_name))
                        self.UI.ini_config_path = os.path.join(os.path.abspath(self.UI.pickfolder),
                            config_name + ".config")
                    else:
                        self.UI.ini_config_path = os.path.join(os.path.abspath(self.UI.pickfolder),
                            config_name + ".config")
                    self.UI.Text_SavePath.setText(os.path.join(self.UI.pickfolder, config_name))
                else:
                    self.UI.ini_config_path = os.path.join(
                        os.path.abspath(self.UI.pickfolder),
                        config_name + ".config")
                if os.path.exists(self.UI.ini_config_path):
                    self.set_config_all.emit(True)
                    self.set_surface(True)
                    self.set_find(True)
                    # self.UI.Init_config_all()
                    # self.UI.graphicsView.Load_config_surface()
                    # self.UI.Load_config_find()
                else:
                    self.set_config_all.emit(True)
                    #self.UI.Init_config_all()
        except Exception as e:
            self.exception = e
            self.error.emit(True)
