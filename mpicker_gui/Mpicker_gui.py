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

try:
    import open3d as o3d
except:
    print("will not use open3d.")
from PyQt5.QtWidgets import *
from PyQt5 import uic,QtCore
from PyQt5.QtGui import QPixmap,QImage,QFont,QKeySequence
import mrcfile
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance
import os,re,copy,sys
import argparse
import configparser
import math
import json
import glob

from mpicker_framememseg import UI_Mpicker_FrameMemseg
# from Mpicker_v2_help import UI_help
from mpicker_mrcview import Mpicker_MrcView
from mpicker_framesetting import Mpicker_FrameSetting
from mpicker_framefind import Mpicker_FrameFind
from mpicker_frameextract import Mpicker_FrameExtract
from mpicker_resultside import Mpicker_ResultSide
from mpicker_frameresult import Mpicker_FrameResult
from mpicker_item import Cross,Circle
from mpicker_resultview import Mpicker_ResultView
from mpicker_framemanual import Mpicker_FrameManual
from mpicker_core import Get_Boundary
from Mpicker_convert_mrc import read_surface_coord
from mpicker_frameadvance import UI_Mpicker_Advance
from mpicker_classeditor import class_File

from splitter import *
from Mpicker_check import *
from mpicker_checkxy import *
from mpicker_checkyz import *
from mpicker_checkxz import *
import multiprocessing
from typing import List,Dict
for_fake_import = False
if for_fake_import:
    import Mpicker_particles

ui_path = os.path.abspath(os.path.dirname(__file__))
ui_file = os.path.join(ui_path,'imagev2.ui')
Ui_MainWindow, QBaseClass  = uic.loadUiType(ui_file)

'''
Program     : MPicker GUI
Stuff Member: Shudong Li
Date        : 2022/03/15

'''


class UI(QMainWindow,Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)


        # MenuBar Code
        self.advance_setting = UI_Mpicker_Advance()
        self.advance_setting.setParameters(self)
        self.actionAdvanceSetting.triggered.connect(self.openWindow_advance)
        self.actionAbout.triggered.connect(self.openWindow_help)
        self.tabWidget.tabBarClicked.connect(self.tabBarInit)
        # mrc pic process parameter by default
        self.pixmap             = None
        self.realx              = 1
        self.realy              = 1
        self.realz              = 1
        self.tomo_check_show    = None
        self.radius_surf        = 6
        self.tomo_manual_show   = None
        self.tomo_data          = None
        self.tomo_mask          = None
        self.tomo_boundary      = None
        self.tomo_show          = None
        self.tomo_select        = []
        self.tomo_select_surf = []
        self.tomo_select_surf_xyz = []
        self.tomo_select_surf_mode = []
        self.tomo_select_surf_direction = []
        self.minsurf            = []
        self.ad_minsurf         = 10
        self.ad_ncpu            = 1
        self.ad_expandratio     = 0
        self.advance_setting.checkBox_adprinttime.setChecked(False)
        self.tomo_check_current = 0
        self.tomo_current       = 0
        self.tomo_surfcurrent   = 0
        self.tomo_number        = 1
        self.tomo_manual_surf   = []
        self.tomo_manual_select = []
        self.tomo_manual_current= 0
        self.tomo_manual_surfcurrent = 0
        self.tomo_manual_number = 1
        self.pressX             = 1
        self.pressY             = 1
        self.slider_Contrast_value = 40
        self.slider_Contrast_Max= 2550
        self.slider_resultContrast_Max = 2550
        self.slider_Bright_Max  = 2550
        self.slider_Bright_value= 40
        self.tomo_show_std      = 0
        self.tomo_show_mean     = 0
        self.mrc_tomo_max       = 255
        self.mrc_tomo_min       = 0
        self.old_mrc_tomo_max   = 255
        self.old_mrc_tomo_min   = 0
        self.number             = None
        # Show the final mrc
        self.scroll_hbox = QHBoxLayout()
        self.scroll_widget = QWidget()
        self.scroll_widget.mousePressEvent = self.ScrollmousePressEvent
        self.scroll_number = 0
        self.scrollHspacing = 5
        self.scrollVspacing = 5
        # parameter
        self.allow_showMrcImage = True
        self.allow_showResultImage = True
        self.allow_showResultImage_side = True
        self.show_ask       = True
        self.show_ask_surf  = True
        self.show_ask_check = True
        self.select_pic     = None
        self.show_pic       = None
        self.save_path      = None
        self.select_label   = None
        self.select_path    = None
        self.select_pic     = None
        self.select_layout  = None
        self.tomo_result    = None
        self.tomo_addresult = None
        self.resultpixmap   = None
        self.screenshot_Fxy  = None
        self.screenshot_Fyz  = None
        self.resultx        = 1
        self.resulty        = 1
        self.resultz        = 1
        self.label_resultx  = 0
        self.label_resulty  = 0
        self.mrc_contrast   = 0
        self.result_contrast= 0
        self.slider_resultContrast_value = 40
        self.slider_resultBright_value = 40
        self.slider_resultBright_Max = 2550
        self.resize_resultpoint = 1
        self.tomo_result_select : List[Mpicker_particles.ParticleData] = []
        self.tomo_result_select_all : Dict[int,List[Mpicker_particles.ParticleData]] = {1:[]}
        self.class_file_name = ".particle_class_editor.txt"
        self.epicker_config_name = ".epicker.config"
        self.class_file : class_File = class_File(None)
        self.particle_class_edit = 1
        self.particle_class_show = [1]
        #self.tomo_result_back = []
        #self.tomo_result_show = []
        #self.tomo_result_stack = []
        self.tomo_result_current = 0
        self.CoordMapping = None
        # self.EraseFlag = False
        self.savetxt_path = None
        self.tomo_extract = None
        self.MRC_path = None
        self.MASK_path = None
        self.SURF_path = None
        self.ManualTxt_path = None
        self.namesuffix = None
        self.Boundary_path = None
        self.Boundary_input_path = None
        self.Result_path = None
        self.Coord_path = None
        self.showResult_path = None
        self.draw_pix_radius = self.spinBox_radius.value()
        self.draw_pix_cursor = 1
        self.draw_pix_linewidth = 1
        # self.label_Press_Flag = False
        self.draw_changed_Flag = False
        self.select_group = None
        self.select_group_Flag = False
        self.Hide_Surf_Flag = False
        self.Picked_point = [-1,-1,-1]
        self.arrow_angle = [-1]
        self.ini_config = configparser.ConfigParser()
        self.surf_config= configparser.ConfigParser()
        self.extract_config = configparser.ConfigParser()
        self.ini_config_path = ""
        self.surf_config_path = ""
        self.extract_config_path = ""
        self.manual_config  = configparser.ConfigParser()
        self.manual_config_path = ""
        self.pickfolder = None
        self.mutex = QtCore.QMutex()
        self.waitCondition = QtCore.QWaitCondition()
        self.Mask_save_folder = None
        self.Mask_getraw = None
        self.Mask_mode = "seg_process"
        self.Mask_gpuid = "0"
        self.saveoutofbound = False
        self.openmemseg = False
        self.surf_mode = []
        self.surf_xyz = []
        self.surf_direction = []
        self.surf_right_pos = None
        self.result_tomo_min = 0
        self.result_tomo_max = 255
        self.show_mrc_flag = False
        self.show_xyz_flag = False
        self.MainUI_Show    = True
        self.label_ClassEdit : QLabel
        self.label_ClassEdit.setStyleSheet(f"background: silver")

        # Show the select result point
        self.select_result_scroll = self.findChild(QScrollArea, "scrollArea_Select_Result")
        self.select_result_scroll_vbox = QVBoxLayout()
        self.select_result_scroll_widget = QWidget()
        self.select_result_scroll_widget.mousePressEvent = self.ScrollSelectresultpoint_mousePressEvent

        self.scrollArea_auto_Select_vbox = QVBoxLayout()
        self.scrollArea_Select_vbox = QVBoxLayout()
        self.scrollArea_manualsurf_vbox = QVBoxLayout()
        # func
        self.Button_Remove.clicked.connect(self.scroll_remove)
        self.Button_Check.clicked.connect(self.scroll_check)
        self.Button_Next.clicked.connect(self.scroll_Next)

        # Graphic Scene
        self.graphicsScene = QGraphicsScene()
        self.graphicsScene_result = QGraphicsScene()
        self.graphicsScene_resultside = QGraphicsScene()
        # Graphic View
        self.graphicsView : Mpicker_MrcView
        self.graphicsView_result : Mpicker_ResultView
        self.graphicsView_resultside : Mpicker_ResultSide
        self.frame_Setting : Mpicker_FrameSetting
        self.frame_FindSurface : Mpicker_FrameFind
        self.frame_ExtractSurface : Mpicker_FrameExtract
        self.frame_Result : Mpicker_FrameResult
        self.frame_manual : Mpicker_FrameManual
        self.graphicsView.setDragMode(self.graphicsView.ScrollHandDrag)
        self.graphicsView.setParameters(self)
        self.graphicsView_result.setDragMode(self.graphicsView_result.ScrollHandDrag)
        self.graphicsView_result.setParameters(self)
        self.graphicsView_resultside.setDragMode(self.graphicsView_resultside.ScrollHandDrag)
        self.graphicsView_resultside.setParameters(self)
        self.frame_Setting.setParameters(self)
        self.frame_FindSurface.setParameters(self)
        self.frame_ExtractSurface.setParameters(self)
        self.frame_Result.setParameters(self)
        self.frame_manual.setParameters(self)
        #self.frame_memseg.setParameters(self)
        # Graphics view
        self.graphicsView.setTransformationAnchor(self.graphicsView.AnchorUnderMouse)
        self.graphicsView_result.setTransformationAnchor(self.graphicsView_result.AnchorUnderMouse)
        self.graphicsView_resultside.setTransformationAnchor(self.graphicsView_resultside.AnchorUnderMouse)

        self.Init_GUI_fromsys()
        self.graphicsView.Shortcut_AddPoint.activated.connect(self.graphicsView.saveReference)
        self.graphicsView.Shortcut_NextPoint.activated.connect(self.graphicsView.nextPoint)
        self.graphicsView.Shortcut_DeletePoint.activated.connect(self.graphicsView.DeleteReference)
        self.tabBarInit(self.tabWidget.currentIndex())
        self.show()

    def setup_class_file(self, file_dir : str):
        file_name = os.path.join(file_dir, self.class_file_name)
        self.class_file = class_File(file_name)
        self.particle_class_edit = self.class_file.edit_idx()
        self.particle_class_show = self.class_file.show_idxs()
        try:
            # close old class_editor
            self.graphicsView_result.editor.close()
        except:
            pass

    def update_class_file(self):
        if self.class_file.no_file:
            return
        self.class_file.read_file()
        self.particle_class_edit = self.class_file.edit_idx()
        self.particle_class_show = self.class_file.show_idxs()

    @staticmethod
    def save_pixmap(pixmap, pre):
        if pixmap is None:
            return
        idx = 0
        while os.path.exists(f"{pre}_{idx:03}.tif"):
            idx += 1
        fname = os.path.abspath(f"{pre}_{idx:03}.tif")
        pixmap.save(fname, "TIFF")
        print("save image as:", fname)
        return

    def closeEvent(self,event):
        reply = QMessageBox.question(self,
                                         'Closing Mpicker',
                                         "Do you want to close all the windows?",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
        if reply == QMessageBox.Yes:
            QtCore.QCoreApplication.instance().quit()
        else:
            event.ignore()
        # self.memseg.close()
        # self.advance_setting.close()




    def tabBarInit(self,index):
        if index == 0:
            self.tabWidget.setCurrentIndex(index)
            if self.pixmap is not None:
                self.showMrcImage = self.graphicsView.showMrcImage
                # if self.graphicsView.EraseFlag:
                #     self.graphicsView.EraseFlag = False
                #     self.Button_Erase_manual.setStyleSheet("background-color: None")
                # else:
                try:
                    self.graphicsView.customContextMenuRequested.disconnect()
                except TypeError:
                    pass
                self.graphicsView.customContextMenuRequested.connect(self.graphicsView.GraphicViewMenu)
                try:
                    self.graphicsView.Shortcut_AddPoint.activated.disconnect()
                    self.graphicsView.Shortcut_NextPoint.activated.disconnect()
                    self.graphicsView.Shortcut_DeletePoint.activated.disconnect()
                except:
                    pass
                self.graphicsView.Shortcut_AddPoint.activated.connect(self.graphicsView.saveReference)
                self.graphicsView.Shortcut_NextPoint.activated.connect(self.graphicsView.nextPoint)
                self.graphicsView.Shortcut_DeletePoint.activated.connect(self.graphicsView.DeleteReference)
                self.showMrcImage()
        elif index == 1:
            self.tabWidget.setCurrentIndex(index)
            if self.pixmap is not None:
                self.showMrcImage = self.frame_manual.showMrcImage
                # if self.graphicsView.EraseFlag:
                #     self.graphicsView.EraseFlag = False
                #     self.Button_Erase_auto.setStyleSheet("background-color: None")
                # else:
                try:
                    self.graphicsView.customContextMenuRequested.disconnect()
                except TypeError:
                    pass
                self.graphicsView.customContextMenuRequested.connect(self.frame_manual.GraphicViewMenu)
                # Set ShortCut
                try:
                    self.graphicsView.Shortcut_AddPoint.activated.disconnect()
                    self.graphicsView.Shortcut_NextPoint.activated.disconnect()
                    self.graphicsView.Shortcut_DeletePoint.activated.disconnect()
                except:
                    pass
                self.graphicsView.Shortcut_AddPoint.activated.connect(self.frame_manual.saveReference)
                self.graphicsView.Shortcut_NextPoint.activated.connect(self.frame_manual.nextPoint)
                self.graphicsView.Shortcut_DeletePoint.activated.connect(self.frame_manual.DeleteReference)
                self.showMrcImage()
        else:
            self.tabWidget.setCurrentIndex(index)
            if self.pixmap is not None:
            #     if self.graphicsView.EraseFlag:
            #         self.graphicsView.EraseFlag = False
            #         self.Button_Erase_manual.setStyleSheet("background-color: None")
            #     else:
                try:
                    self.graphicsView.customContextMenuRequested.disconnect()
                except TypeError:
                    pass



    def Init_GUI_fromsys(self):
        # config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui.config")
        # if os.path.exists(config_path):
        #     self.init_widgets()
        if "--mask" in sys.argv:
            index_mask = sys.argv.index('--mask')
            self.MASK_path =  os.path.abspath(sys.argv[index_mask + 1])
            self.tomo_boundary = Get_Boundary(self.MASK_path)
            self.frame_Setting.actionMask.setEnabled(True)
            self.frame_Setting.actionBoundary.setEnabled(True)
            with mrcfile.mmap(self.MASK_path) as mrc:
                self.tomo_mask = mrc.data
                self.tomo_mask = self.tomo_mask[:, ::-1, :]
                # tomo_min = np.min(self.tomo_mask)
                # tomo_max = np.max(self.tomo_mask)
                # self.tomo_mask = ((self.tomo_mask - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                self.tomo_show = self.tomo_mask
                self.show_mrc_flag = False
                self.Init_MrcImage()

        if "--raw" in sys.argv:
            index_raw = sys.argv.index('--raw')
            self.MRC_path = os.path.abspath(sys.argv[index_raw + 1])
            self.frame_Setting.actionRaw.setEnabled(True)
            with mrcfile.mmap(self.MRC_path) as mrc:
                self.tomo_data = mrc.data[:, ::-1, :]
                # tomo_min = np.min(self.tomo_data)
                # tomo_max = np.max(self.tomo_data)
                # self.tomo_data = ((self.tomo_data - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                self.tomo_show = self.tomo_data
                self.show_mrc_flag = True
                self.Init_MrcImage()

        if "--out" in sys.argv:
            index_out = sys.argv.index('--out')
            self.Text_SavePath.setText(sys.argv[index_out + 1])
            self.pickfolder = sys.argv[index_out + 1]

        if "--config" in sys.argv:
            index_out = sys.argv.index("--config")
            config_path = sys.argv[index_out + 1]
            if os.path.exists(config_path):
                self.ini_config.read(config_path, encoding='utf-8')
                self.ini_config_path = config_path
                self.pickfolder = os.path.abspath(os.path.dirname(config_path))
                self.frame_Setting.Thread_Openconfig.setParameters(self)
                self.frame_Setting.Thread_Openconfig.start()
                self.frame_Setting.disabled_all_button()
                #self.Init_config()
            else:
                print("no this file")

        self.frame_Setting.finished_Open()


    def Init_config(self):
        if self.ini_config.has_section("Path"):
            if self.ini_config.get('Path', 'InputRaw') != "None":
                self.MRC_path = self.ini_config.get('Path', 'InputRaw')
                self.frame_Setting.actionRaw.setEnabled(True)
                with mrcfile.mmap(self.MRC_path) as mrc:
                    self.tomo_data = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.tomo_data)
                    # tomo_max = np.max(self.tomo_data)
                    # self.tomo_data = ((self.tomo_data - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    self.tomo_show = self.tomo_data
                    self.show_mrc_flag = True
                    self.Init_MrcImage()
            else:
                self.MRC_path = None
            if self.ini_config.get('Path', 'InputMask') != "None":
                self.MASK_path = self.ini_config.get('Path',"InputMask")
                self.frame_Setting.actionMask.setEnabled(True)
                with mrcfile.mmap(self.MASK_path) as mrc:
                    self.tomo_mask = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.tomo_mask)
                    # tomo_max = np.max(self.tomo_mask)
                    # self.tomo_mask = ((self.tomo_mask - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
            else:
                self.MRC_path = None
            if self.ini_config.get('Path', 'InputBoundary') != "None":
                self.Boundary_path = self.ini_config.get('Path', "InputBoundary")
                self.Text_SavePath.setText(os.path.dirname(self.Boundary_path))
                self.frame_Setting.actionBoundary.setEnabled(True)
                with mrcfile.mmap(self.Boundary_path) as mrc:
                    self.tomo_boundary = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.tomo_boundary)
                    # tomo_max = np.max(self.tomo_boundary)
                    # if tomo_max != tomo_min:
                    #     self.tomo_boundary = ((self.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    # else:
                    #     self.tomo_boundary = np.zeros(self.tomo_boundary.shape)
                self.show_mrc_flag = False
            else:
                self.Boundary_path = None


    def Init_config_all(self):
        if self.Text_SavePath.toPlainText() != "" and self.MRC_path is not None:
            config_name = os.path.splitext(os.path.basename(self.MRC_path))[0]
            if self.tomo_boundary is not None:
                self.Boundary_path = os.path.join(self.Text_SavePath.toPlainText(),'my_boundary_6.mrc')
                self.Boundary_input_path = self.Boundary_path
                with mrcfile.new(self.Boundary_path, overwrite=True) as mrc:
                    tomo_boundary = self.tomo_boundary[:, ::-1, :].astype(np.int8)
                    tomo_min = np.min(tomo_boundary).astype(float)
                    tomo_max = np.max(tomo_boundary).astype(float)
                    if tomo_max != tomo_min:
                        tomo_boundary = (tomo_boundary - tomo_min) / (tomo_max - tomo_min)
                    else:
                        tomo_boundary = np.zeros(tomo_boundary.shape)
                    #tomo_boundary = ((tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    mrc.set_data(tomo_boundary.astype(np.int8))
                with mrcfile.mmap(self.Boundary_path) as mrc:
                    self.tomo_boundary = mrc.data[:, ::-1, :]

            if os.path.exists(self.ini_config_path) is False:
                config_file = open(self.ini_config_path, "w+").close()
            self.ini_config.read(self.ini_config_path, encoding='utf-8')
            # First Time Set up
            if self.ini_config.has_section("Path") is False:
                self.ini_config.add_section("Path")
                self.ini_config.set('Path', "InputRaw", str(self.MRC_path))
                self.ini_config.set('Path', "InputMask", str(self.MASK_path))
                self.ini_config.set('Path', 'InputBoundary', str(self.Boundary_path))
                self.ini_config.set('Path', "Surface", "None")
            else:
                if self.MRC_path is not None:
                    self.ini_config.set('Path', "InputRaw", str(os.path.abspath(self.MRC_path)))
                if self.MASK_path is not None:
                    self.ini_config.set('Path', "InputMask", str(os.path.abspath(self.MASK_path)))
                    self.ini_config.set('Path', 'InputBoundary', str(os.path.abspath(self.Boundary_path)))
                if self.ini_config.get('Path',"Surface") != "None":
                    Surface_string = self.ini_config.get('Path', 'Surface')
                    Surface_path   = self.Text_SavePath.toPlainText()#,os.path.join(config_name)
                    #os.path.dirname(self.ini_config.get('Path', 'Inputboundary'))
                    #print("Surface_path = ",Surface_path)
                    Surface_list = Surface_string.split(" ")
                    Surface_new_string = ""
                    for Surface_str in Surface_list:
                        path = os.path.join(Surface_path, Surface_str)
                        #print("path = ",path)
                        if os.path.exists(path):
                            #print("True")
                            Surface_new_string = Surface_new_string + " " + Surface_str
                    if Surface_new_string == "":
                        self.ini_config.set('Path', 'Surface', "None")
                    else:
                        self.ini_config.set('Path', 'Surface', Surface_new_string)

            with open(self.ini_config_path,"w") as config_file:
                self.ini_config.write(config_file)

    def Load_config_find(self):
        if self.ini_config.get("Path","Surface") != "None":
            folder_path = self.Text_SavePath.toPlainText()#os.path.dirname(self.ini_config.get("Path","InputBoundary"))
            surf_list   = self.ini_config.get("Path","Surface").split(" ")
            for surf_one in surf_list:
                surf_path = os.path.join(folder_path, surf_one)
                all_file = os.listdir(surf_path)
                valid_file = []
                temp_file = None
                for file in all_file:
                    if "_result.mrc" in file:
                        configname = file.split("_")[0] + file.split("_")[1] + ".config"
                        if configname in all_file:
                            valid_file.append(file)
                if len(valid_file) != 0:
                    for i in range(len(valid_file)):
                        extract_i = int(valid_file[i].split("_")[1].split("-")[1])
                        extract_min = np.inf
                        extract_num = 0
                        for j in range(i + 1, len(valid_file)):
                            extract_j = int(valid_file[j].split("_")[1].split("-")[1])
                            if extract_min > extract_j:
                                extract_min = extract_j
                                extract_num = j
                        if extract_i > extract_min:
                            temp_file = valid_file[i]
                            valid_file[i] = valid_file[extract_num]
                            valid_file[extract_num] = temp_file
                    for file in valid_file:
                        #self.UI.Result_path = os.path.join(surf_path, file)
                        #self.set_extract_surf.emit(file)
            # for surf_one in surf_list:
            #     surf_path   = os.path.join(folder_path,surf_one)
            #     all_file    = os.listdir(surf_path)
            #     for file in all_file:
            #         if "_result.mrc" in file:
                        self.Result_path = os.path.join(surf_path,file)
                        chose_index = file.split("_")[1]
                        if file.split("_")[0] == "manual":
                            chose_index = "M"+chose_index
                        each_vbox = QVBoxLayout()
                        Name_label = QLabel()
                        Name_label.setText(chose_index)
                        Name_label.setScaledContents(True)
                        Font = QFont("Agency FB", 10)
                        Font.setBold(True)
                        Name_label.setFont(Font)
                        Path_label = QLabel()
                        Path_label.setText(self.Result_path)
                        Path_label.setHidden(True)
                        with mrcfile.mmap(self.Result_path) as mrc:
                            self.tomo_addresult = mrc.data[:, ::-1, :]
                            # tomo_min = np.min(self.tomo_addresult)
                            # tomo_max = np.max(self.tomo_addresult)
                            # self.tomo_addresult = ((self.tomo_addresult - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                        mrc_image_Image = self.UI.tomo_addresult[int(self.UI.tomo_addresult.shape[0] / 2), :, :]
                        tomo_min = np.min(mrc_image_Image).astype(float)
                        tomo_max = np.max(mrc_image_Image).astype(float)
                        mrc_image_Image = ((mrc_image_Image - tomo_min) / (tomo_max - tomo_min) * 255)
                        Pro_label = QLabel()
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
                        each_vbox.setSpacing(self.scrollVspacing)
                        Pro_label.setToolTip(self.Result_path)
                        QToolTip.setFont(QFont('Agency FB', 10))
                        self.scroll_hbox.addLayout(each_vbox)
                        self.scroll_hbox.setSpacing(self.scrollHspacing)
                        self.scroll_widget.setLayout(self.scroll_hbox)
                        self.scrollArea.setWidget(self.scroll_widget)

    def openWindow_help(self):
        #self.window = QMainWindow()
        #self.ui     = UI_help()
        # self.memseg = UI_Mpicker_FrameMemseg()
        # self.memseg.setParameters(self)
        # self.memseg.show()
        #self.memseg.setParameters(self)
        return

    def openWindow_advance(self):
        self.advance_setting.show()

    def Init_Scroll_Mrc(self, auto_surfcurrent=None):
        if self.tabWidget.currentWidget().objectName() == "tab_auto":
            chosed_label = self.scrollArea_Select_vbox.itemAt(self.tomo_surfcurrent).layout().itemAt(0).widget()
            chosed_number = chosed_label.text().split(". ")[0]
        elif self.tabWidget.currentWidget().objectName() == "tab_manual":
            chosed_label = self.scrollArea_manualsurf_vbox.itemAt(self.tomo_manual_surfcurrent).layout().itemAt(0).widget()
            chosed_number = "M"+chosed_label.text().split(". ")[0]
        else:
            pass

        if auto_surfcurrent is not None:
            # because surfcurrent we want may be different from "current" surf.
            auto, surfcurrent = auto_surfcurrent
            if auto:
                chosed_label = self.scrollArea_Select_vbox.itemAt(surfcurrent).layout().itemAt(0).widget()
                chosed_number = chosed_label.text().split(". ")[0]
            else: #manual
                chosed_label = self.scrollArea_manualsurf_vbox.itemAt(surfcurrent).layout().itemAt(0).widget()
                chosed_number = "M"+chosed_label.text().split(". ")[0]

        self.make_number = 0
        for i in range(self.scroll_hbox.count()):
            chosed_label = self.scroll_hbox.itemAt(i).itemAt(0).widget()
            number = chosed_label.text().split("-")[0]
            make_number = int(chosed_label.text().split("-")[1])
            if number == chosed_number and self.make_number <= make_number:
                self.make_number = make_number
        self.make_number = self.make_number + 1

        for i in range(0,self.scroll_hbox.count()):
            if self.scroll_hbox.itemAt(i) is not None:
                chosed_label = self.scroll_hbox.itemAt(i)
                path_label   = chosed_label.itemAt(2).widget()
                if path_label.text() == self.Result_path:
                    self.make_number = int(self.scroll_hbox.itemAt(i).itemAt(0).widget().text().split("-")[1])
                    self.scroll_hbox.itemAt(i).itemAt(0).widget().deleteLater()
                    self.scroll_hbox.itemAt(i).itemAt(1).widget().deleteLater()
                    self.scroll_hbox.itemAt(i).itemAt(2).widget().deleteLater()
                    #self.scroll_hbox.removeItem(self.scroll_hbox.itemAt(i))

        each_vbox      = QVBoxLayout()
        Name_label = QLabel()
        Name_label.setText(chosed_number+"-"+str(self.make_number))
        Name_label.setScaledContents(True)
        Font = QFont("Agency FB",10)
        Font.setBold(True)
        Name_label.setFont(Font)
        Path_label = QLabel()
        Path_label.setText(self.Result_path)
        Path_label.setHidden(True)
        # setlabel
        Pro_label = QLabel()
        mrc_image_Image = self.tomo_addresult[int(self.tomo_addresult.shape[0]/2),:,:]
        tomo_min = np.min(mrc_image_Image).astype(float)
        tomo_max = np.max(mrc_image_Image).astype(float)
        mrc_image_Image = (mrc_image_Image - tomo_min) / (tomo_max - tomo_min) * 255
        mrc_image_Image = Image.fromarray(mrc_image_Image).convert('RGB')
        mrc_image_Image = np.asarray(mrc_image_Image)
        # Show the Image
        height, width, channels = mrc_image_Image.shape
        bytesPerLine = channels * width
        qImg = QImage(mrc_image_Image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = QPixmap.fromImage(qImg)
        Pro_label.setPixmap(qImg)
        Pro_label.setScaledContents(True)
        wh_ratio    = qImg.width()/qImg.height()
        Pro_height  = 220
        Pro_width   = int(Pro_height * wh_ratio)
        Pro_label.setFixedSize(Pro_width , Pro_height)

        each_vbox.addWidget(Name_label)
        each_vbox.addWidget(Pro_label)
        each_vbox.addWidget(Path_label)
        each_vbox.setSpacing(self.scrollVspacing)

        Pro_label.setToolTip(self.Result_path)
        QToolTip.setFont(QFont('Agency FB', 10))

        self.scroll_hbox.addLayout(each_vbox)
        self.scroll_hbox.setSpacing(self.scrollHspacing)
        self.scroll_widget.setLayout(self.scroll_hbox)
        self.scrollArea.setWidget(self.scroll_widget)

    def scroll_Next(self):
        if self.select_path is not None:
            if self.save_path is None:
                self.Text_Information.clear()
            else:
                self.Set_Original_Information()
            for i in range(0,self.scroll_hbox.count()):
                chosed_label = self.scroll_hbox.itemAt(i).layout()
                piclabel = chosed_label.itemAt(1).widget()
                namelabel = chosed_label.itemAt(0).widget()
                if piclabel == self.select_pic:
                    if self.showResult_path != self.select_path.text():
                        piclabel.setStyleSheet("")
                        namelabel.setStyleSheet("")
                    if i + 1 < self.scroll_hbox.count():
                        i_next = i + 1
                        # next_label = self.scroll_hbox.itemAt(i+1).layout()
                    else:
                        i_next = 0
                        # next_label = self.scroll_hbox.itemAt(0).layout()
                    next_label = self.scroll_hbox.itemAt(i_next).layout()
                    self.select_pic = next_label.itemAt(1).widget()
                    self.select_label = next_label.itemAt(0).widget()
                    self.select_path = next_label.itemAt(2).widget()
                    self.select_layout = self.scroll_hbox.itemAt(i_next)
                    if self.select_path.text() == self.showResult_path:
                        self.select_label.setStyleSheet("QLabel{color : orange;}")
                        self.select_pic.setStyleSheet("border: 6px solid orange;")
                    else:
                        self.select_label.setStyleSheet("QLabel{color : orange;}")
                        self.select_pic.setStyleSheet("border: 3px solid orange;")
                    self.Text_SavePath.setText(
                            os.path.dirname(os.path.abspath(os.path.dirname(self.select_path.text()))))
                    self.Set_Text_Information()
                    self.Load_config_extract()
                    break
            self.scroll_check()


    def ScrollSelectresultpoint_mousePressEvent(self,event):
        if self.resultpixmap is not None:
            if event.buttons() & QtCore.Qt.LeftButton:
                x = event.x()
                y = event.y()
                tag_nochange = True
                for i in range(self.select_result_scroll_vbox.count()):
                    Font = QFont("Agency FB", 9)
                    chosed_label = self.select_result_scroll_vbox.itemAt(i).widget()
                    pic_xmin, pic_ymin = chosed_label.x(), chosed_label.y()
                    pic_xmax, pic_ymax = chosed_label.x() + chosed_label.width(), chosed_label.y() + chosed_label.height()
                    if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                        try:
                            Font.setBold(True)
                            self.tomo_result_current = i #int(re.findall(r"(.*): ", chosed_label.text())[0]) - 1
                            self.resultx = self.tomo_result_select[self.tomo_result_current].x
                            self.resulty = self.tomo_result_select[self.tomo_result_current].y
                            self.resultz = self.tomo_result_select[self.tomo_result_current].z
                        except:
                            pass
                    else:
                        Font.setBold(False)
                    chosed_label.setFont(Font)
                if tag_nochange:
                    # clicked no label
                    i = self.tomo_result_current
                    if self.select_result_scroll_vbox.itemAt(i) is not None:
                        chosed_label = self.select_result_scroll_vbox.itemAt(i).widget()
                        Font = QFont("Agency FB", 9)
                        Font.setBold(True)
                        chosed_label.setFont(Font)
                self.showResultImage()


    def scroll_check(self):
        if self.select_layout is not None and self.select_path is not None:
            if self.showResult_path != self.select_path.text():
                self.graphicsScene_result.clear()
                self.graphicsScene_resultside.clear()
                self.graphicsScene_result = QGraphicsScene()
                self.graphicsScene_resultside = QGraphicsScene()
                for i in range(0, self.scroll_hbox.count()):
                    self.scroll_hbox.itemAt(i).layout().itemAt(1).widget().setStyleSheet("")
                    self.scroll_hbox.itemAt(i).layout().itemAt(0).widget().setStyleSheet("")
                self.select_label.setStyleSheet("QLabel{color : orange;}")
                self.select_pic.setStyleSheet("border: 6px solid orange;")
                self.show_pic = self.select_pic
                self.save_path = self.select_path.text()
                self.tomo_result_select = []
                self.tomo_result_select_all = {1:[]}
                #self.tomo_result_back = []
                self.resultpixmap = None
                self.resultx = 1
                self.resulty = 1
                self.resultz = 1
                self.tomo_result_current = 0
                # load the new mrc
                self.showResult_path    = self.select_path.text()
                with mrcfile.mmap(self.showResult_path) as mrc:
                    self.tomo_result = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.tomo_result)
                    # tomo_max = np.max(self.tomo_result)
                    # self.tomo_result = ((self.tomo_result - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                self.CoordMapping = np.load(self.showResult_path.replace("_result.mrc", "_convert_coord.npy"))
                self.allow_showMrcImage = False
                self.Init_MrcResultImage()
                self.showResultImage()
                self.allow_showMrcImage = True # in fact it will already be True
                self.namesuffix = os.path.basename(self.showResult_path).split("-")[0]
                self.result_folder = os.path.abspath(os.path.dirname(self.showResult_path))
                # clear the select protein scroll area
                for del_i in reversed(range(self.select_result_scroll_vbox.count())):
                    self.select_result_scroll_vbox.itemAt(del_i).widget().setParent(None)
                if self.tabWidget.currentWidget().objectName() == "tab_auto":
                    if self.pixmap is not None:
                        self.SURF_path = self.showResult_path.replace(os.path.basename(self.showResult_path),self.namesuffix + "_surf.mrc.npz")
                        # self.tomo_extract = read_surface_mrc(self.SURF_path, dtype=bool)[:, ::-1, :]
                        self.tomo_extract = read_surface_coord(self.SURF_path)
                        # self.tomo_extract = (self.tomo_extract * 255).astype(np.uint8)
                        # with mrcfile.open(self.SURF_path) as mrc:
                        #     self.tomo_extract = mrc.data[:, ::-1, :]
                            # tomo_min = np.min(self.tomo_extract)
                            # tomo_max = np.max(self.tomo_extract)
                            # self.tomo_extract = ((self.tomo_extract - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    config_path = os.path.join(self.result_folder, os.path.basename(self.showResult_path).split("-")[0] + ".config")
                    self.surf_config.read(config_path, encoding='utf-8')
                    for i in range(self.scrollArea_Select_vbox.count()):
                        chosed_label = self.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()
                        if chosed_label.text().split(". ")[0] == os.path.basename(self.showResult_path).split("-")[0].split("_")[1]:
                            self.tomo_surfcurrent = i
                            break
                    self.tomo_check_current = self.tomo_surfcurrent
                    self.tomo_select    = copy.deepcopy(self.tomo_select_surf[self.tomo_surfcurrent])
                    self.surf_mode      = copy.deepcopy(self.tomo_select_surf_mode[self.tomo_surfcurrent])
                    self.surf_xyz       = copy.deepcopy(self.tomo_select_surf_xyz[self.tomo_surfcurrent])
                    self.surf_direction = copy.deepcopy(self.tomo_select_surf_direction[self.tomo_surfcurrent])
                    self.tomo_current = 0
                    self.realx = self.tomo_select[self.tomo_current][0]
                    self.realy = self.tomo_select[self.tomo_current][1]
                    self.realz = self.tomo_select[self.tomo_current][2]
                    self.Load_config_extract()
                    #self.graphicsView.Reset_surf_show_parameter()
                    self.graphicsView.Reset_scrollArea_Select()
                    self.graphicsView.Reset_surf_show_parameter()
                    self.graphicsView.Reset_scrollArea_Select_points()
                    self.graphicsView_result.LoadresultPoint()
                    self.frame_manual.Reset_scrollArea_Surf_Select()
                    self.showMrcImage()
                elif self.tabWidget.currentWidget().objectName() == "tab_manual":
                    if self.pixmap is not None:
                        self.ManualTxt_path = self.showResult_path.replace(os.path.basename(self.showResult_path),
                                                                           self.namesuffix + "_surf.txt")
                        for i in range(self.scrollArea_manualsurf_vbox.count()):
                            chosed_label = self.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(0).widget()
                            if chosed_label.text().split(". ")[0] == \
                                    os.path.basename(self.showResult_path).split("-")[0].split("_")[1]:
                                self.tomo_manual_surfcurrent = i
                        self.tomo_manual_select = self.tomo_manual_surf[self.tomo_manual_surfcurrent]
                        self.tomo_check_current = self.tomo_manual_surfcurrent
                        self.tomo_manual_current = 0
                        self.realx              = self.tomo_manual_select[self.tomo_manual_current][0]
                        self.realy              = self.tomo_manual_select[self.tomo_manual_current][1]
                        self.realz              = self.tomo_manual_select[self.tomo_manual_current][2]
                        self.frame_manual.Reset_scrollArea_Select()
                        self.frame_manual.scrollArea_manualpoints_widget.setLayout(self.frame_manual.scrollArea_manualpoints_vbox)
                        self.frame_manual.scrollArea_manualpoints.setWidget(self.frame_manual.scrollArea_manualpoints_widget)
                        self.frame_manual.Reset_scrollArea_Surf_Select()
                        self.frame_manual.showMrcImage()
                        self.graphicsView_result.LoadresultPoint()
                        self.graphicsView.Reset_scrollArea_Select()


    def Load_config_extract(self):
        #print("self.result_folder (Load config)= ",self.result_folder)# Original=  self.showResult_path
        # Now
        if self.select_label.text()[0] != "M":
            config_path = os.path.join(os.path.dirname(self.select_path.text()),
                                       "surface_"+os.path.basename(self.select_path.text()).split("_")[1] + ".config")
        else:
            config_path = os.path.join(os.path.dirname(self.select_path.text()),
                                       "manual_"+os.path.basename(self.select_path.text()).split("_")[1] + ".config")
        #print("config path = ",config_path)
        self.extract_config.read(config_path, encoding='utf-8')
        extractmode = self.extract_config.get('Parameter','method')
        if extractmode == self.radioButton_choseRBF.text():
            self.radioButton_choseRBF.click()
        elif extractmode == self.radioButton_chosePOLY.text():
            self.radioButton_chosePOLY.click()
        self.spinBox_RBFsample.setValue(self.extract_config.getfloat('Parameter','RBFsample'))
        self.spinBox_Order.setValue(self.extract_config.getint('Parameter','PolyOrder'))
        self.spinBox_Cylinderorder.setValue(self.extract_config.getint('Parameter','cylinderorder'))
        if self.extract_config.getfloat('Parameter', 'smoothfactor') > 9:
            value = len(self.extract_config.get('Parameter', 'smoothfactor'))
            self.doubleSpinBox_Smoothfactor.setValue(value)
        else:
            self.doubleSpinBox_Smoothfactor.setValue(self.extract_config.getfloat('Parameter','smoothfactor'))
        self.spinBox_Thickness.setValue(self.extract_config.getint('Parameter','thickness'))
        self.doubleSpinBox_Fillvalue.setValue(self.extract_config.getfloat('Parameter','fillvalue'))


    def Init_MrcResultImage(self):
        self.resultz                        = int(round(self.tomo_result.shape[0] / 2))
        self.resultx                        = int(round(self.tomo_result.shape[2] / 2))
        self.resulty                        = int(round(self.tomo_result.shape[1] / 2))

        # reset the origin
        self.old_result_tomo_max            = np.max(self.tomo_result).astype(float)
        self.old_result_tomo_min            = np.min(self.tomo_result).astype(float)
        old_result_tomo_all                 = self.old_result_tomo_max - self.old_result_tomo_min
        self.tomo_result_std                = float(np.std(self.tomo_result)) # round(np.std(self.tomo_result), 1)
        if self.tomo_result_std == 0:
            self.tomo_result_std = 0.1
        self.tomo_result_mean               = float(np.mean(self.tomo_result)) # round(np.mean(self.tomo_result), 1)
        self.slider_resultContrast_Max      = round(math.ceil(old_result_tomo_all / np.std(self.tomo_result)) * 10)
        self.slider_resultBright_Max        = round(self.slider_resultContrast_Max / 2)
        self.slider_resultContrast_value    = round(self.slider_resultContrast_Max * 1 / 2 / 10,1) #10
        self.slider_resultBright_value      = 0
        self.result_tomo_min                = self.tomo_result_mean + (self.slider_resultBright_value -
                                              self.slider_resultContrast_value) * self.tomo_result_std
        self.result_tomo_max                = self.tomo_result_mean + (self.slider_resultBright_value +
                                              self.slider_resultContrast_value) * self.tomo_result_std
        self.result_contrast                = 255 / (self.result_tomo_max - self.result_tomo_min)
        # savetxtpath
        self.savetxt_path = self.showResult_path.replace("_result.mrc", "_SelectPoints.txt")
        if os.path.isfile(self.savetxt_path) == False:
            makefile = open(self.savetxt_path, "w").close()
        self.horizontalSlider_resultContrast.setMinimum(1)
        self.horizontalSlider_resultContrast.setMaximum(self.slider_resultContrast_Max)
        self.horizontalSlider_resultContrast.setValue(self.slider_resultContrast_value * 10)
        self.doubleSpinBox_resultContrast.setValue(self.slider_resultContrast_value)
        self.horizontalSlider_resultBright.setMinimum(-self.slider_resultBright_Max/2)
        self.horizontalSlider_resultBright.setMaximum(self.slider_resultBright_Max/2)
        self.horizontalSlider_resultBright.setValue(self.slider_resultBright_value * 10)
        self.doubleSpinBox_resultBright.setValue(self.slider_resultBright_value)

        self.doubleSpinBox_resultContrast.setMinimum(0.1)
        self.doubleSpinBox_resultContrast.setMaximum(self.slider_resultContrast_Max / 10)
        self.doubleSpinBox_resultContrast.setValue(self.slider_resultContrast_value)
        self.doubleSpinBox_resultBright.setMinimum(-self.slider_resultBright_Max / 10)
        self.doubleSpinBox_resultBright.setMaximum(self.slider_resultBright_Max / 10)
        self.doubleSpinBox_resultBright.setValue(self.slider_resultBright_value)

        # reset the slider_z
        self.resulthorizontalSlider_z.setMinimum(1)
        self.resulthorizontalSlider_z.setMaximum(self.tomo_result.shape[0])
        self.resulthorizontalSlider_z.setValue(self.resultz)
        self.doubleSpinBox_resultX.setMinimum(1)
        self.doubleSpinBox_resultX.setMaximum(self.tomo_result.shape[2])
        self.doubleSpinBox_resultX.setValue(self.resultx)
        self.doubleSpinBox_resultY.setMinimum(1)
        self.doubleSpinBox_resultY.setMaximum(self.tomo_result.shape[1])
        self.doubleSpinBox_resultY.setValue(self.resulty)
        self.doubleSpinBox_resultZ.setMinimum(1)
        self.doubleSpinBox_resultZ.setMaximum(self.tomo_result.shape[0])
        self.doubleSpinBox_resultZ.setValue(self.resultz)
        self.spinBox_resultz.setMinimum(1)
        self.spinBox_resultz.setMaximum(self.tomo_result.shape[0])
        self.spinBox_resultz.setValue(self.resultz)
        # reset the xyz label
        self.x_resultpos_label.setText(f"/{self.tomo_result.shape[2]}")
        self.y_resultpos_label.setText(f"/{self.tomo_result.shape[1]}")
        self.z_resultpos_label.setText(f"/{self.tomo_result.shape[0]}")
        self.label_resultzlimit.setText(f"/{self.tomo_result.shape[0]}")
        self.label_resultcontrastlimit.setText(f"/{self.slider_resultContrast_Max/10}")
        self.label_resultbrightlimit.setText(f"/{self.slider_resultBright_Max/10}")
        # Zproject
        self.spinBox_Zproject.setValue(0)
        # reset view
        self.graphicsScene_resultside = QGraphicsScene()
        self.graphicsScene_result = QGraphicsScene()

    def scroll_remove(self):
        if self.select_layout is not None and self.select_path is not None:
            if os.access(self.select_path.text(), os.W_OK):
                if self.show_ask is False:
                    self.Remove_scroll_widget()
                else:
                    self.scroll_remove_popupInfo()
            else:
                QMessageBox.warning(self, "Access Warning",
                                    "Could not access to the data.\n Path = "
                                    + self.select_path.text() +
                                    "Could not remove the file"
                                    , QMessageBox.Ok)


    def Remove_scroll_widget(self):
        Remove_id   = self.select_label.text()
        Remove_path = self.select_path.text()
        self.select_pic.setParent(None)
        self.select_label.setParent(None)
        self.select_path.setParent(None)
        self.select_layout.setParent(None)
        self.Text_Information.clear()
        # self.select_pic = None
        # self.select_label = None
        # self.select_path = None
        # self.scroll_hbox.removeItem(self.select_layout)
        if self.showResult_path != self.select_path.text():
            pass
        else:
            for del_i in reversed(range(self.select_result_scroll_vbox.count())):
                self.select_result_scroll_vbox.itemAt(del_i).widget().setParent(None)
            self.show_pic = None
            self.save_path = None
            self.resultpixmap = None
            self.sidepixmap = None
            self.tomo_result = None
            self.showResult_path = None
            self.graphicsView.Reset_scrollArea_Select()
            self.frame_manual.Reset_scrollArea_Surf_Select()
            # self.tomo_extract = None
            self.graphicsScene_result.clear()
            self.graphicsScene_resultside.clear()
            self.graphicsScene_result = QGraphicsScene()
            self.graphicsScene_resultside = QGraphicsScene()
        self.select_label = None
        self.select_pic = None
        self.select_layout = None
        self.select_path = None
        if Remove_id[0] != "M":
            folder_name = os.path.dirname(Remove_path)
            Remove_config = os.path.join(folder_name, "surface_" + Remove_id + ".config")
        else:
            folder_name = os.path.dirname(Remove_path)
            manual_id = "manual_" + Remove_id[1:]
            Remove_config = os.path.join(folder_name, manual_id + ".config")
        # Remove_npy_path = Remove_path.replace("result.mrc", "convert_coord.npy")
        # Remove_points = Remove_path.replace("result.mrc", "SelectPoints.txt")
        # Remove_rbfcoords = Remove_config.replace( ".config", "_RBF_InterpCoords.txt")
        Remove_pre = Remove_path[:Remove_path.rfind("result.mrc")]
        Remove_npy_path = Remove_pre + "convert_coord.npy"
        Remove_points = Remove_pre + "SelectPoints.txt"
        Remove_epicker_pre = Remove_pre + "epicker*"
        Remove_rbfcoords = Remove_config[:Remove_config.rfind(".config")] + "_RBF_InterpCoords.txt"
        if os.path.exists(Remove_path):
            self.tomo_addresult = None # close mmap
            os.remove(Remove_path)
        if os.path.exists(Remove_config):
            os.remove(Remove_config)
        if os.path.exists(Remove_npy_path):
            os.remove(Remove_npy_path)
        if os.path.exists(Remove_points):
            os.remove(Remove_points)
        if os.path.exists(Remove_rbfcoords):
            os.remove(Remove_rbfcoords)
        for file in glob.glob(Remove_epicker_pre):
            if os.path.isfile(file):
                os.remove(file)

    def scroll_remove_popupInfo(self):
        msg = QMessageBox()
        msg.setWindowTitle("Delete Warning")
        msg.setText('The Select Result '+ self.select_label.text() +' will be deteled!\n'
                    'Do you still want to proceed? ')
        #msg.setInformativeText("Team Name Must Not Be Empty Or Contain Any Special Characters!")
        msg.setIcon(QMessageBox.Warning)
        cb = QCheckBox()
        cb.setText("Don't show this again")
        msg.setCheckBox(cb)
        msg.setWindowTitle("Delete Warning")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        final = msg.exec_()
        if final == QMessageBox.Yes:
            self.Remove_scroll_widget()
            if cb.isChecked():
                self.show_ask = False
        else:
            if cb.isChecked():
                self.show_ask = False

    def ScrollmousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            x = event.x()
            y = event.y()
            if self.save_path is None:
                self.Text_Information.clear()
            else:
                self.Set_Original_Information()
            for i in range(0,self.scroll_hbox.count()):
                chosed_label = self.scroll_hbox.itemAt(i).layout()
                piclabel = chosed_label.itemAt(1).widget()
                namelabel = chosed_label.itemAt(0).widget()
                if piclabel != self.show_pic:
                    piclabel.setStyleSheet("")
                    namelabel.setStyleSheet("")
            for i in range(0,self.scroll_hbox.count()):
                chosed_label = self.scroll_hbox.itemAt(i).layout()
                piclabel = chosed_label.itemAt(1).widget()
                namelabel = chosed_label.itemAt(0).widget()
                if piclabel != self.show_pic:
                    piclabel.setStyleSheet("")
                    namelabel.setStyleSheet("")
                pic_xmin,pic_ymin = piclabel.x(),piclabel.y()
                pic_xmax,pic_ymax = piclabel.x() + piclabel.width(),piclabel.y() + piclabel.height()
                if x > pic_xmin and x < pic_xmax and y > pic_ymin and y < pic_ymax:
                    if piclabel != self.show_pic:
                        if self.select_pic != piclabel:
                            self.select_pic   = chosed_label.itemAt(1).widget()
                            self.select_label = chosed_label.itemAt(0).widget()
                            self.select_path  = chosed_label.itemAt(2).widget()
                            self.select_layout = self.scroll_hbox.itemAt(i)
                            self.select_pic.setStyleSheet("border: 3px solid orange;")
                            self.select_label.setStyleSheet("QLabel { color : orange;}")
                            self.Text_SavePath.setText(os.path.dirname(os.path.abspath(os.path.dirname(self.select_path.text()))))
                            self.Set_Text_Information()
                            self.Load_config_extract()
                        else:
                            self.select_label.setStyleSheet("QLabel {color : black}")
                            self.select_pic.setStyleSheet("")
                            #self.Set_Original_Information()
                            self.select_pic     = None
                            self.select_label   = None
                            self.select_path    = None
                            self.select_layout  = None
                    else:
                        self.select_pic = chosed_label.itemAt(1).widget()
                        self.select_label = chosed_label.itemAt(0).widget()
                        self.select_path = chosed_label.itemAt(2).widget()
                        self.select_layout = self.scroll_hbox.itemAt(i)
                        self.Set_Text_Information()
                        self.Load_config_extract()
                        #self.Load_config_find()
                    break
                self.select_pic = None
                self.select_label = None
                self.select_path = None
                self.select_layout = None


    def Set_Original_Information(self):
        config_name = os.path.basename(self.save_path).split("_")[1]
        config_type = os.path.basename(self.save_path).split("_")[0]
        config_path = os.path.join(os.path.dirname(self.save_path), config_type +"_"+config_name + ".config")
        if config_type == "manual":
            self.tabBarInit(1)
        else:
            self.tabBarInit(0)
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        Text = os.path.basename(self.save_path)
        if config.has_section("Parameter"):
            Text = Text + "\nmethod = " + config.get("Parameter", "method")
            Text = Text + "\nrbfsample = " + config.get("Parameter", "rbfsample")
            Text = Text + "\npolyorder =" + config.get("Parameter", "polyorder")
            Text = Text + "\nthickness = " + config.get("Parameter", "thickness")
            Text = Text + "\ncylinderorder = " + config.get("Parameter", "cylinderorder")
            Text = Text + "\nfillvalue = " + config.get("Parameter", "fillvalue")
            if float(config.get("Parameter", "smoothfactor")) > 9:
                Text = Text + "\nsmoothfactor = " + str(len(config.get("Parameter", "smoothfactor")))
            else:
                Text = Text + "\nsmoothfactor = " + config.get("Parameter", "smoothfactor")
            self.Text_Information.setText(Text)

    def Set_Text_Information(self):
        if self.select_label is not None:
            config_name = os.path.basename(self.select_path.text()).split("_")[1]
            config_type = os.path.basename(self.select_path.text()).split("_")[0]
            config_path = os.path.join(os.path.dirname(self.select_path.text()),config_type+"_"+config_name+".config")
            if config_type == "manual":
                self.tabBarInit(1)
            else:
                self.tabBarInit(0)
            config      = configparser.ConfigParser()
            config.read(config_path, encoding='utf-8')
            Text = os.path.basename(self.select_path.text())
            if config.has_section("Parameter"):
                Text    = Text + "\nmethod = " + config.get("Parameter","method")
                Text    = Text + "\nrbfsample = " + config.get("Parameter", "rbfsample")
                Text    = Text + "\npolyorder =" + config.get("Parameter", "polyorder")
                Text    = Text + "\nthickness = " + config.get("Parameter", "thickness")
                Text    = Text + "\ncylinderorder = " + config.get("Parameter", "cylinderorder")
                Text    = Text + "\nfillvalue = " + config.get("Parameter", "fillvalue")
                if float(config.get("Parameter", "smoothfactor")) > 9:
                    Text    = Text + "\nsmoothfactor = " + str(len(config.get("Parameter", "smoothfactor")))
                else:
                    Text = Text + "\nsmoothfactor = " + config.get("Parameter", "smoothfactor")
                self.Text_Information.setText(Text)



    def Init_MrcImage(self):
        self.realz                  = int(round(self.tomo_show.shape[0] / 2))
        self.realx                  = int(round(self.tomo_show.shape[2] / 2))
        self.realy                  = int(round(self.tomo_show.shape[1] / 2))

        # self.mrc_image              = self.tomo_show[self.realz-1, :, :]
        # reset the origin
        try:
            with mrcfile.open(self.tomo_show.filename) as mrc:
                # calculate min max of memmap might be very slow.
                tomo = mrc.data
            self.old_mrc_tomo_max       = np.max(tomo).astype(float)
            self.old_mrc_tomo_min       = np.min(tomo).astype(float)
            old_mrc_all_range           = self.old_mrc_tomo_max - self.old_mrc_tomo_min
            self.tomo_show_std          = float(np.std(tomo)) # round(np.std(tomo),1)
            self.tomo_show_mean         = float(np.mean(tomo)) # round(np.mean(tomo),1)
            del tomo
        except:
            self.old_mrc_tomo_max       = np.max(self.tomo_show).astype(float)
            self.old_mrc_tomo_min       = np.min(self.tomo_show).astype(float)
            old_mrc_all_range           = self.old_mrc_tomo_max - self.old_mrc_tomo_min
            self.tomo_show_std          = float(np.std(self.tomo_show)) # round(np.std(self.tomo_show),1)
            self.tomo_show_mean         = float(np.mean(self.tomo_show)) # round(np.mean(self.tomo_show),1)
        if self.tomo_show_std == 0:
            self.tomo_show_std = 0.1
        if self.show_mrc_flag:
            self.slider_Contrast_Max    = round(math.ceil(old_mrc_all_range / self.tomo_show_std) * 10)
            self.slider_Bright_Max      = round(self.slider_Contrast_Max / 2)
            if self.tomo_show_std == 0.1:
                self.slider_Contrast_value = int(self.slider_Contrast_Max / 2 / 10)
            else:
                self.slider_Contrast_value  = int(self.slider_Contrast_Max/4/10)
            self.slider_Bright_value    = 0
            self.mrc_tomo_min           = self.tomo_show_mean + (self.slider_Bright_value  -
                                                                 self.slider_Contrast_value) * self.tomo_show_std
            self.mrc_tomo_max           = self.tomo_show_mean + (self.slider_Bright_value  +
                                                                 self.slider_Contrast_value) * self.tomo_show_std
            self.mrc_contrast           = 255 / (self.mrc_tomo_max - self.mrc_tomo_min)
        # reset the slider_contrast
        self.horizontalSlider_Contrast.setMinimum(1)
        self.horizontalSlider_Contrast.setMaximum(self.slider_Contrast_Max)
        self.horizontalSlider_Contrast.setValue(self.slider_Contrast_value*10)
        self.horizontalSlider_Bright.setMinimum(-self.slider_Bright_Max)
        self.horizontalSlider_Bright.setMaximum(self.slider_Bright_Max)
        self.horizontalSlider_Bright.setValue(self.slider_Bright_value*10)
        self.doubleSpinBox_Contrast.setMinimum(0.1)
        self.doubleSpinBox_Contrast.setMaximum(self.slider_Contrast_Max/10)
        self.doubleSpinBox_Contrast.setValue(self.slider_Contrast_value)
        self.doubleSpinBox_Bright.setMinimum(-self.slider_Bright_Max/10)
        self.doubleSpinBox_Bright.setMaximum(self.slider_Bright_Max/10)
        self.doubleSpinBox_Bright.setValue(self.slider_Bright_value)
        # reset the slider_z
        self.horizontalSlider_z.setMinimum(1)
        self.horizontalSlider_z.setMaximum(self.tomo_show.shape[0])
        self.horizontalSlider_z.setValue(round(self.tomo_show.shape[0] / 2))
        self.doubleSpinBox_X.setMinimum(1)
        self.doubleSpinBox_X.setMaximum(self.tomo_show.shape[2])
        self.doubleSpinBox_Y.setMinimum(1)
        self.doubleSpinBox_Y.setMaximum(self.tomo_show.shape[1])
        self.doubleSpinBox_Z.setMinimum(1)
        self.doubleSpinBox_Z.setMaximum(self.tomo_show.shape[0])
        # reset the xyz label
        self.doubleSpinBox_X.setValue(self.realx)
        self.doubleSpinBox_Y.setValue(self.realy)
        self.doubleSpinBox_Z.setValue(self.realz)
        self.x_pos_label.setText(f"/{self.tomo_show.shape[2]}")
        self.y_pos_label.setText(f"/{self.tomo_show.shape[1]}")
        self.z_pos_label.setText(f"/{self.tomo_show.shape[0]}")
        self.label_zlimit.setText(f"/{self.tomo_show.shape[0]}")
        self.label_contrastlimit.setText(f"/{self.slider_Contrast_Max/10}")
        self.label_brightlimit.setText(f"/{self.slider_Bright_Max/10}")

        # line edit z
        self.spinBox_z.setValue(self.realz)
        # self.graphicsView.update()
        self.graphicsScene = QGraphicsScene()
        # self.graphicsScene.addItem(self.Cursor)
        self.graphicsView.setScene(self.graphicsScene)
        self.showMrcImage()


    def showMrcImage(self, fromResult=False):
        '''
        reload the func showMrcImage() in mpicker_mrcview.py or mpicker_framemanual.py
        fromResult means it is called by showResultImage
        '''
        pass

    def showResultImage(self):
        '''
        reload the func showResultImage() in mpicker_resultview.py
        '''
        pass

    def removeResultCursor(self):
        '''
        reload the func removeResultCursor() in mpicker_resultview.py
        '''
        pass

    def showResultImage_side(self, hidecross=False):
        '''
        reload the func showResultImage_side() in mpicker_resultside.py
        '''
        pass

    def removeResultCursor_side(self):
        '''
        reload the func removeResultCursor_side() in mpicker_resultside.py
        '''
        pass



if __name__ == "__main__":
    multiprocessing.freeze_support() # add this when pyinstaller on Windows
    multiprocessing.set_start_method('spawn') # to save memory
    parser = argparse.ArgumentParser(description="Mpicker GUI Shudong Li, 20220227")
    parser.add_argument('--mask'    , type = str, default = ''  , help = 'path to mask mrc file')
    parser.add_argument('--raw'     , type = str, default = ''  , help = 'path to raw mrc file')
    parser.add_argument('--out'     , type = str, default = ''  , help = 'path to save all the results')
    parser.add_argument('--config'  , type = str, default = ''  , help = 'path to config file')
    # config parser
    # end
    args = parser.parse_args()
    # QApplication.setStyle(QStyleFactory.create("Fusion")) # QStyleFactory::keys()
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # os.putenv("QT_FONT_DPI", "96")
    app = QApplication(sys.argv)
    #MainWindow = QMainWindow()
    UIWindow = UI()
    sys.exit(app.exec_())


# # BACK UP
# if Remove_id[0] != "M":
#     folder_name = os.path.dirname(Remove_path)
#     Remove_config = os.path.join(folder_name, "surface_" + Remove_id + ".config")
#     Remove_npy_path = Remove_path.replace("result.mrc", "convert_coord.npy")
#     Remove_points = Remove_path.replace("result.mrc", "SelectPoints.txt")
#     if os.path.exists(Remove_path):
#         print("Remove", Remove_path)
#     # else:
#     #     QMessageBox.warning(self, "Remove Error",
#     #                         "There is no such '_result.mrc' file "
#     #                         "in " + os.path.dirname(Remove_path)
#     #                         , QMessageBox.Ok)
#     if os.path.exists(Remove_config):
#         print("Remove", Remove_config)
#     # else:
#     #     QMessageBox.warning(self, "Remove Error",
#     #                         "There is no such '.config' file "
#     #                         "in " + os.path.dirname(Remove_config)
#     #                         , QMessageBox.Ok)
#     if os.path.exists(Remove_npy_path):
#         print("Remove", Remove_npy_path)
#     # else:
#     #     QMessageBox.warning(self, "Remove Error",
#     #                         "There is no such _result.npy file "
#     #                         "in " + os.path.dirname(Remove_npy_path)
#     #                         , QMessageBox.Ok)
#     if os.path.exists(Remove_points):
#         print("Remove", Remove_points)
#     # os.remove(Remove_path)
# else:
#     print("Manual")
#     folder_name = os.path.dirname(Remove_path)
#     manual_id = "manual_" + Remove_id[1:]
#     Remove_config = os.path.join(folder_name, manual_id + ".config")
#     Remove_npy_path = Remove_path.replace("result.mrc", "convert_coord.npy")
#     Remove_points = Remove_path.replace("result.mrc", "SelectPoints.txt")
#     if os.path.exists(Remove_path):
#         print("Remove", Remove_path)
#     # else:
#     #     QMessageBox.warning(self, "Remove Error",
#     #                         "There is no such '_result.mrc' file "
#     #                         "in " + os.path.dirname(Remove_path)
#     #                         , QMessageBox.Ok)
#     if os.path.exists(Remove_config):
#         print("Remove", Remove_config)
#     # else:
#     #     QMessageBox.warning(self, "Remove Error",
#     #                         "There is no such '.config' file "
#     #                         "in " + os.path.dirname(Remove_config)
#     #                         , QMessageBox.Ok)
#     if os.path.exists(Remove_npy_path):
#         print("Remove", Remove_npy_path)
#     # else:
#     #     QMessageBox.warning(self, "Remove Error",
#     #                         "There is no such _result.npy file "
#     #                         "in " + os.path.dirname(Remove_npy_path)
#     #                         , QMessageBox.Ok)
#     if os.path.exists(Remove_points):
#         print("Remove", Remove_points)