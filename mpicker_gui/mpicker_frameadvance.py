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

from PyQt5.QtWidgets import     QMainWindow,QFileDialog,\
                                QMenu,QMessageBox,\
                                QVBoxLayout,QWidget,QLabel,QFrame,\
                                QShortcut
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5 import uic,QtCore
import os,sys,time
import argparse
import numpy as np
import mrcfile
from Mpicker_check import Mpicker_check

# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class UI_Mpicker_Advance(QMainWindow):
    def __init__(self):
        super(UI_Mpicker_Advance, self).__init__()
        # Load the ui file
        self.uifile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advanceSetting.ui")
        uic.loadUi(self.uifile_path, self)

    def setParameters(self, UI):
        self.UI = UI
        #self.spinBox_adminsurf.setValue(self.UI.ad_minsurf)
        self.spinBox_adminsurf.valueChanged.connect(self.change_ad_minsurf)
        #self.spinBox_adncpu.setValue(self.UI.ad_ncpu)
        self.spinBox_adncpu.valueChanged.connect(self.change_ad_ncpu)
        #self.doubleSpinBox_adexpandratio.setValue(self.UI.ad_expandratio)
        self.doubleSpinBox_adexpandratio.valueChanged.connect(self.change_ad_expandratio)
        # self.Mask_mode = None
        # self.Mask_getraw = None
        # self.show_mrc   = None
        # self.show_name  = None
        # #self.Button_Savemask.clicked.connect(self.clicker_SetMaskSavePath)
        # self.Button_Memseg.clicked.connect(self.clicker_GetMask)
        # self.Button_multiMemseg.clicked.connect(self.clicker_MultiGetMask)
        # #self.Button_Postprocess.clicked.connect(self.clicker_PostProcess)
        # self.Button_Selectinputfolder.clicked.connect(self.clicker_GetInputfolder)
        # self.Button_check.clicked.connect(self.clicker_check)
        # self.lineEdit_gpu.editingFinished.connect(self.lineEdit_gpu_finishededit)

    def change_ad_minsurf(self):
        self.UI.ad_minsurf      = self.spinBox_adminsurf.value()

    def change_ad_ncpu(self):
        self.UI.ad_ncpu         = self.spinBox_adncpu.value()

    def change_ad_expandratio(self):
        self.UI.ad_expandratio  = self.doubleSpinBox_adexpandratio.value()
