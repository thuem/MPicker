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
from PyQt5.QtCore import QThread,pyqtSignal,QObject,QMutex
import numpy as np
import mrcfile
import os, sys
# import importlib ##change
# import mpicker_core
# from mpicker_core import Find_Surface
import Mpicker_core_gui
from Mpicker_convert_mrc import read_surface_coord
import time
import warnings


class Mpicker_FrameFind(QFrame):
    def __init__(self, *__args):
        super(Mpicker_FrameFind, self).__init__(*__args)
        self.progress = 0

    def setParameters(self, UI):
        self.UI = UI
        self.UI.Button_FindSurface.clicked.connect(self.clicked_Find_Surface)

    def clicked_Find_Surface(self):
        if self.UI.MRC_path is not None:
            if self.UI.MASK_path is not None:
                if len(self.UI.tomo_select) != 0:
                    if self.UI.Text_SavePath.toPlainText() != "":
                        self.progress = 0
                        self.UI.progressBar.setValue(0)
                        self.UI.label_progress.setText("Finding Surface...Please Wait")
                        self.UI.Button_FindSurface.setEnabled(False)
                        self.UI.Button_ExtractSurface.setEnabled(False)
                        self.Thread_find_surface = QThread_Find()
                        self.Thread_find_surface.setParameters(self.UI)
                        self.Thread_find_surface.finished.connect(self.Reset_Button)
                        self.Thread_find_surface.load_progress.connect(self.load_progress)
                        self.Thread_find_surface.error_signal.connect(self.raiseerror)
                        self.UI.graphicsView.saveSurfReference()
                        self.Thread_find_surface.start()
                    else:
                        QMessageBox.warning(self, "Output Warning", "Please Select the output directory\n", QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, "Input Warning", "Please Select Points before find surface\n", QMessageBox.Ok)
            else:
                QMessageBox.warning(self, "Input Warning", "Please Input a Mask before find surface\n", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "Input Warning", "Please Input a Raw before find surface\n", QMessageBox.Ok)

    def raiseerror(self,bool):
        if bool:
            str_e = repr(self.Thread_find_surface.exception)
            QMessageBox.warning(self, "Point picking Error", str_e, QMessageBox.Ok)
            self.Thread_find_surface.terminate()

    def load_progress(self, val):
        while self.progress < val:
            self.progress += 1
            self.UI.progressBar.setValue(self.progress)

    def Reset_Button(self):
        self.UI.showMrcImage()
        self.UI.label_progress.setText("Find Surface Done")
        self.UI.Button_FindSurface.setEnabled(True)
        self.UI.Button_ExtractSurface.setEnabled(True)
        #self.Thread_find_surface.kill_thread()



class QThread_Find(QThread):
    def __init__(self):
        super(QThread_Find,self).__init__()
        self.exitcode = False
        self.exception = None

    load_progress   = pyqtSignal(int)
    error_signal    = pyqtSignal(bool)
    def setParameters(self, UI):
        self.UI = UI

    def run(self):
        self.set_SavePath()
        self.load_progress.emit(50)

        core_path = Mpicker_core_gui.__file__
        cmd = f'{sys.executable} {core_path} --mode surffind --config_tomo {self.UI.ini_config_path} --config_surf {self.UI.surf_config_path}'
        print(f'\033[0;35m{cmd}\033[0m')

        # # parameters settings
        # left2right = []
        # for direction in self.UI.surf_direction:
        #     if direction == "Left To Right":
        #         left2right.append(True)
        #     else:
        #         left2right.append(False)

        # try:
        #     # importlib.reload(mpicker_core) ##change
        #     # from mpicker_core import Find_Surface, read_surface_mrc
        #     Find_Surface(boundary_path  = self.UI.Boundary_input_path,
        #                     mask_path       = self.UI.MASK_path,
        #                     initial_point   = self.UI.tomo_select_surf[self.UI.tomo_surfcurrent],#self.UI.tomo_select[self.UI.tomo_current]
        #                     surfout_path    = self.UI.SURF_path,
        #                     boundaryout_path= self.UI.Boundary_path,
        #                     near_ero        = self.UI.comboBox_Nearerode.currentText(),
        #                     left2right      = left2right,
        #                     xyz             = self.UI.surf_xyz,
        #                     surf_method     = self.UI.surf_mode,
        #                     min_surf        = self.UI.ad_minsurf,#spinBox_minsu,rf.value(),
        #                     elongation_pixel= self.UI.spinBox_maxpixel.value(),
        #                     n_cpu           = self.UI.ad_ncpu
        #                  )

        try:
            s = os.system(cmd)
            if s != 0:
                print("exit code", s)
                raise Exception("surface finding failed, see terminal for details")
            self.load_progress.emit(75)
            
            self.UI.tomo_extract = read_surface_coord(self.UI.SURF_path)
            if self.UI.Boundary_input_path is None:
                with mrcfile.mmap(self.UI.Boundary_path) as mrc:
                    self.UI.tomo_boundary = mrc.data[:, ::-1, :]
            else:
                with mrcfile.mmap(self.UI.Boundary_input_path) as mrc:
                    self.UI.tomo_boundary = mrc.data[:, ::-1, :]
            self.UI.Picked_point = self.UI.tomo_select[self.UI.tomo_current]

        except Exception as e:
            self.exitcode = True
            self.exception = e
            self.error_signal.emit(True)

        # if self.exitcode is False:
            # self.load_progress.emit(75)
            # self.UI.tomo_extract = read_surface_mrc(self.UI.SURF_path, dtype=bool)[:, ::-1, :]
            # self.UI.tomo_extract = read_surface_coord(self.UI.SURF_path)
            # tomo_min = np.min(self.UI.tomo_extract)
            # tomo_max = np.max(self.UI.tomo_extract)
            # self.UI.tomo_extract = ((self.UI.tomo_extract - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
            # if self.UI.Boundary_input_path is None:
            #     with mrcfile.mmap(self.UI.Boundary_path) as mrc:
            #         self.UI.tomo_boundary = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.UI.tomo_boundary)
                    # tomo_max = np.max(self.UI.tomo_boundary)
                    # if tomo_max != tomo_min:
                    #     self.UI.tomo_boundary = (
                    #             (self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    # else:
                    #     self.UI.tomo_boundary = np.zeros(self.UI.tomo_boundary.shape).astype(np.uint8)
                    #self.UI.show_mrc_flag = False
                    #self.UI.tomo_boundary = ((self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
            # else:
            #     with mrcfile.mmap(self.UI.Boundary_input_path) as mrc:
            #         self.UI.tomo_boundary = mrc.data[:, ::-1, :]
                    # tomo_min = np.min(self.UI.tomo_boundary)
                    # tomo_max = np.max(self.UI.tomo_boundary)
                    # if tomo_max != tomo_min:
                    #     self.UI.tomo_boundary = (
                    #             (self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
                    # else:
                    #     self.UI.tomo_boundary = np.zeros(self.UI.tomo_boundary.shape).astype(np.uint8)
                    #self.UI.show_mrc_flag = False
                    #self.UI.tomo_boundary = ((self.UI.tomo_boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)

            # self.UI.Picked_point = self.UI.tomo_select[self.UI.tomo_current]
            # if self.UI.comboBox_xyz.currentText() == 'x':
            #     if left2right is False:
            #         self.UI.arrow_angle = 180
            #     else:
            #         self.UI.arrow_angle = 0
            # elif self.UI.comboBox_xyz.currentText() == 'y':
            #     if left2right:
            #         self.UI.arrow_angle = 90
            #     else:
            #         self.UI.arrow_angle = 270
            # else:
            #     self.UI.arrow_angle = -1
        self.load_progress.emit(100)

    def set_SavePath(self):
        # create result folder
        chosed_label = self.UI.scrollArea_Select_vbox.itemAt(self.UI.tomo_surfcurrent).layout().itemAt(0).widget()
        chosed_number = chosed_label.text().split(". ")[0]
        # self.UI.str_xyz = "_x" + str(round(self.UI.tomo_select[self.UI.tomo_current][0])) + "_y" + str(
        #     round(self.UI.tomo_select[self.UI.tomo_current][1])) + "_z" + str(
        #     round(self.UI.tomo_select[self.UI.tomo_current][2]))
        self.UI.result_folder = os.path.join(self.UI.Text_SavePath.toPlainText(),"surface_"+
                                             chosed_number+"_"+os.path.splitext(os.path.basename(self.UI.MRC_path))[0])
        if os.path.exists(self.UI.result_folder) == False:
            os.mkdir(self.UI.result_folder)
        chosed_label = self.UI.scrollArea_Select_vbox.itemAt(self.UI.tomo_surfcurrent).layout().itemAt(0).widget()
        chosed_number = chosed_label.text().split(". ")[0]
        self.UI.namesuffix        = "surface_"+chosed_number #+"_"+self.UI.comboBox_Surfmode.currentText()+"_"+self.UI.comboBox_xyz.currentText()+"_nearero"+self.UI.comboBox_Nearerode.currentText() +"_"
        self.UI.SURF_path      = os.path.join(self.UI.result_folder,self.UI.namesuffix + "_surf.mrc.npz")
        self.UI.Boundary_path  = os.path.join(self.UI.Text_SavePath.toPlainText() , 'my_boundary_' + self.UI.comboBox_Nearerode.currentText() + '.mrc')
        if os.path.exists(self.UI.Boundary_path):
            self.UI.Boundary_input_path = self.UI.Boundary_path
        else:
            self.UI.Boundary_input_path = None
        # self.Init_config_surface()
        self.surf_name = self.UI.namesuffix + "_" + os.path.splitext(os.path.basename(self.UI.MRC_path))[0]
        self.Init_config_find()
        # self.SaveCoord_path = self.result_folder

    # def Init_config_surface(self):
    #     self.UI.ini_config.read(self.UI.ini_config_path, encoding='utf-8')
    #     if self.UI.ini_config.has_section("Path"):
    #         if self.UI.ini_config.get('Path','Surface') == "None":
    #             Surface_string = self.UI.namesuffix + "_" + os.path.splitext(os.path.basename(self.UI.MRC_path))[0]
    #             self.UI.ini_config.set('Path', 'Surface', Surface_string)
    #         else:
    #             same_flag = False
    #             Surface_string = self.UI.ini_config.get('Path', 'Surface')
    #             for Surface in Surface_string.split(" "):
    #                 if Surface ==  self.UI.namesuffix + "_" + os.path.splitext(os.path.basename(self.UI.MRC_path))[0]:
    #                     same_flag = True
    #             if same_flag == False:
    #                 Surface_string = Surface_string + " " +  self.UI.namesuffix + "_" + os.path.splitext(os.path.basename(self.UI.MRC_path))[0]
    #                 self.UI.ini_config.set('Path', 'Surface', Surface_string)
    #     with open(self.UI.ini_config_path, "w") as config_file:
    #         self.UI.ini_config.write(config_file)
    #     self.Init_config_find()

    def Init_config_find(self):
        self.UI.surf_config_path = os.path.join(self.UI.result_folder, self.UI.namesuffix + ".config")
        if os.path.exists(self.UI.surf_config_path) is False:
            config_file = open(self.UI.surf_config_path, "w+")
        self.UI.surf_config.read(self.UI.surf_config_path, encoding='utf-8')
        if self.UI.surf_config.has_section("Parameter") is False:
            self.UI.surf_config.add_section("Parameter")
        self.UI.surf_config.set("Parameter", "ID",self.UI.namesuffix.split("_")[1])
        self.UI.surf_config.set("Parameter", "Points",str(self.UI.tomo_select_surf[self.UI.tomo_surfcurrent]))
        self.UI.surf_config.set("Parameter", "mode", str(self.UI.surf_mode))
        self.UI.surf_config.set("Parameter", "Facexyz", str(self.UI.surf_xyz))
        self.UI.surf_config.set("Parameter", "NearEro", self.UI.comboBox_Nearerode.currentText())
        self.UI.surf_config.set("Parameter", "DirectionL2R", str(self.UI.surf_direction))
        self.UI.surf_config.set("Parameter", "minsurf", str(self.UI.ad_minsurf))#self.UI.spinBox_minsurf.value()
        self.UI.surf_config.set("Parameter", "ncpu",str(self.UI.ad_ncpu))
        self.UI.surf_config.set("Parameter", "maxpixel", str(self.UI.spinBox_maxpixel.value()))
        warnings.filterwarnings("ignore")
        with open(self.UI.surf_config_path, "w") as config_file:
            self.UI.surf_config.write(config_file)
        warnings.filterwarnings("default")


    def kill_thread(self):
        self.terminate()


