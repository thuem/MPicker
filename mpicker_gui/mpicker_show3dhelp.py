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
from mpicker_plot3d import get_help_image

# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class UI_Mpicker_Show3dhelp(QMainWindow):
    def __init__(self):
        super(UI_Mpicker_Show3dhelp, self).__init__()
        # Load the ui file
        #self.uifile_path = resource_path("show3d_help.ui")
        self.uifile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "show3d_help.ui")
        uic.loadUi(self.uifile_path, self)
        self.show()
        help_text, array_img_text = get_help_image()
        self.label.setText(help_text)

    def setParameters(self, UI):
        self.UI = UI