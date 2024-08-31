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
try:
    # from Mpicker_memseg import main_seg, main_post, main_finetune_pre, main_finetune_train
    import Mpicker_memseg
    import torch
    memseg_path = Mpicker_memseg.__file__
except:
    memseg_path = None


def main_seg(input, output, model, gpuid="0", batch=2, thread=4):
    if memseg_path is None:
        return
    cmd = f'{sys.executable} {memseg_path} --mode seg --input_mrc {input} --output_mrc {output} ' \
        + f'--input_model {model} --gpuid {gpuid} --batch {batch} --thread {thread}'
    print(f'\033[1;35m{cmd}\033[0m')
    s = os.system(cmd)
    if s != 0:
        print("exit code", s)
        raise Exception("segmentation failed")
def main_post(input, output, thres, gauss, voxel_cut):
    if memseg_path is None:
        return
    cmd = f'{sys.executable} {memseg_path} --mode post --input_mrc {input} --output_mrc {output} ' \
        + f'--thres {thres: .2f} --gauss {gauss: .2f} --voxel_cut {int(voxel_cut): d}'
    print(f'\033[1;35m{cmd}\033[0m')
    s = os.system(cmd)
    if s != 0:
        print("exit code", s)
        raise Exception("post process failed")
def main_finetune_pre(data, mask, output, z_range="-1,-1"):
    # crop_size="300,300", stride="200,200"
    if memseg_path is None:
        return
    cmd = f'{sys.executable} {memseg_path} --mode finetune_pre --input_mrc {data} --input_mask {mask} ' \
        + f'--output_finetune {output} --z_range={z_range}'
    print(f'\033[1;35m{cmd}\033[0m')
    s = os.system(cmd)
    if s != 0:
        print("exit code", s)
        raise Exception("finetune preparation failed")
def main_finetune_train(data, dataset, model=None, gpuid="0", thread=4):
    if memseg_path is None:
        return
    cmd = f'{sys.executable} {memseg_path} --mode finetune_train --input_mrc {data} --dataset {dataset} ' \
        + f'--input_model {model} --gpuid {gpuid} --thread {thread}'
    print(f'\033[1;35m{cmd}\033[0m')
    s = os.system(cmd)
    if s != 0:
        print("exit code", s)
        raise Exception("finetune failed")


# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class UI_Mpicker_FrameMemseg(QMainWindow):
    def __init__(self):
        super(UI_Mpicker_FrameMemseg, self).__init__()
        # Load the ui file
        self.uifile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memseg.ui")
        uic.loadUi(self.uifile_path, self)
        # self.show()

    def setParameters(self, UI):
        self.parentPath = UI.Text_SavePath.toPlainText()
        self.mrcPath = UI.MRC_path
        if UI.MASK_path is not None:
            if os.path.isfile(UI.MASK_path):
                self.Text_InputMask.setText(UI.MASK_path)

        self.radioButton_defaultM.toggled.connect(self.defaultModel_toggle)
        self.Button_selectModel.clicked.connect(self.selectModel)
        self.Button_runMemseg.clicked.connect(self.run_memseg)
        self.Button_selectMask_ft.clicked.connect(self.selectMask)
        self.radioButton_defaultM_ft.toggled.connect(self.defaultModel_ft_toggle)
        self.Button_selectModel_ft.clicked.connect(self.selectModel_ft)
        self.Button_runFinetune.clicked.connect(self.run_finetune)

        self.Text_progress.setText("No Progress Running")
        self.Thread_memseg = None
        self.Thread_finetune = None

        self.updateParameters()

    def updateParameters(self):
        def text2int(lineEdit, default):
            # return int
            try:
                r = int(lineEdit.text())
            except:
                r = default
                lineEdit.setText(str(default))
            return r
        def text2ints(lineEdit, default):
            # return str
            try:
                r = lineEdit.text().strip().split(",")
                r = [int(i) for i in r]
                if len(r)==0:
                    lineEdit.setText(default)
                    return default
                return lineEdit.text()
            except:
                lineEdit.setText(default)
                return default

        self.gpuid = text2ints(self.lineEdit_gpuid, "0")
        gpu_num = len(self.gpuid.strip().split(","))
        if gpu_num > self.spinBox_batchsize.value():
            self.spinBox_batchsize.setValue(gpu_num)
        self.batch = self.spinBox_batchsize.value()
        self.threads = self.spinBox_threads.value()
        self.id = self.lineEdit_id.text().replace(" ", "")
        if self.radioButton_defaultM.isChecked():
            self.model = None
        else:
            self.model = self.Text_InputModel.toPlainText()
            if self.model == '':
                self.model = None
        self.thres = self.doubleSpinBox_threshold.value()
        self.gauss = self.doubleSpinBox_gauss.value()
        self.voxel = self.spinBox_voxel.value()

        self.gpuid_ft = text2ints(self.lineEdit_gpuid_ft, "0")
        self.threads_ft = self.spinBox_threads_ft.value()
        self.id_ft = self.lineEdit_id_ft.text().replace(" ", "")
        self.z_min = text2int(self.lineEdit_zmin, -1)
        self.z_max = text2int(self.lineEdit_zmax, -1)
        self.maskpath = self.Text_InputMask.toPlainText()
        if self.radioButton_defaultM_ft.isChecked():
            self.model_ft = None
        else:
            self.model_ft = self.Text_InputModel_ft.toPlainText()
            if self.model_ft == '':
                self.model_ft = None

    def defaultModel_toggle(self, checked):
        if checked:
            self.Text_InputModel.setEnabled(False)
            self.Button_selectModel.setEnabled(False)
        else:
            self.Text_InputModel.setEnabled(True)
            self.Button_selectModel.setEnabled(True)

    def defaultModel_ft_toggle(self, checked):
        if checked:
            self.Text_InputModel_ft.setEnabled(False)
            self.Button_selectModel_ft.setEnabled(False)
        else:
            self.Text_InputModel_ft.setEnabled(True)
            self.Button_selectModel_ft.setEnabled(True)

    def selectModel(self):
        fname = QFileDialog.getOpenFileName(self, "Open Model File", self.parentPath, "PTH Files (*.pth);;All Files (*)")[0]
        if fname != '':
            self.Text_InputModel.setText(fname)

    def selectModel_ft(self):
        fname = QFileDialog.getOpenFileName(self, "Open Model File", self.parentPath, "PTH Files (*.pth);;All Files (*)")[0]
        if fname != '':
            self.Text_InputModel_ft.setText(fname)

    def selectMask(self):
        memseg_path = os.path.join(self.parentPath, "memseg")
        if os.path.isdir(memseg_path):
            startpath = memseg_path
        else:
            startpath = self.parentPath
        fname = QFileDialog.getOpenFileName(self, "Open Mask File", startpath, "MRC Files (*.mrc);;All Files (*)")[0]
        if fname != '':
            self.Text_InputMask.setText(fname)

    def update_progress(self, progress):
        self.Text_progress.setText("Running " + progress + "...\nSee terminal for details")

    def show_error(self, progress, err):
        QMessageBox.warning(self, progress+" Error", 
                            err + "\nSee terminal for details" + "\nYou may want to delete tmp file by hand", 
                            QMessageBox.Ok)

    def run_memseg(self):
        self.updateParameters()
        if self.parentPath != '' and self.mrcPath is not None and self.mrcPath != '':
            self.Text_progress.setText("Start Running")
            self.Button_runMemseg.setEnabled(False)
            self.Thread_memseg = QThread_memseg()
            self.Thread_memseg.setParameters(self)
            self.Thread_memseg.finished.connect(self.memseg_finish)
            self.Thread_memseg.progress_signal.connect(self.update_progress)
            self.Thread_memseg.error_signal.connect(self.show_error)
            self.Thread_memseg.start()

    def memseg_finish(self):
        self.Text_progress.setText("No Progress Running")
        self.Button_runMemseg.setEnabled(True)
        self.Thread_memseg.deleteLater()
        self.Thread_memseg = None

    def run_finetune(self):
        self.updateParameters()
        if not os.path.exists(self.maskpath):
            QMessageBox.warning(self, "Input Error", "You must provide a mask file to finetune", QMessageBox.Ok)
            return
        if self.parentPath != '' and self.mrcPath is not None and self.mrcPath != '':
            self.Text_progress.setText("Start Running")
            self.Button_runFinetune.setEnabled(False)
            self.Thread_finetune = QThread_finetune()
            self.Thread_finetune.setParameters(self)
            self.Thread_finetune.finished.connect(self.finetune_finish)
            self.Thread_finetune.progress_signal.connect(self.update_progress)
            self.Thread_finetune.error_signal.connect(self.show_error)
            self.Thread_finetune.start()

    def finetune_finish(self):
        self.Text_progress.setText("No Progress Running")
        self.Button_runFinetune.setEnabled(True)
        self.Thread_finetune.deleteLater()
        self.Thread_finetune = None

    def closeEvent(self,event):
        if self.Text_progress.toPlainText() != "No Progress Running":
            reply = QMessageBox.question(self, 
                                        'Progress is running',
                                        "The Progress may not stop, do you want to exit?",
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()

    
class QThread_memseg(QThread):
    def __init__(self):
        super(QThread_memseg, self).__init__()

    def setParameters(self, mUI:UI_Mpicker_FrameMemseg):
        self.input_raw = mUI.mrcPath
        self.gpuid = mUI.gpuid
        self.batch = mUI.batch
        self.threads = mUI.threads
        self.id = mUI.id
        self.input_model = mUI.model
        self.thres = mUI.thres
        self.gauss = mUI.gauss
        self.voxel_cut = mUI.voxel
        self.skip = mUI.radioButton_skip.isChecked()
        self.parent_path = os.path.join(mUI.parentPath, "memseg") # fix path
        input_mask_name = f"seg_raw_id{self.id}.mrc" # fix name
        self.input_mask = os.path.join(self.parent_path, input_mask_name)
        output_mask_name = f"seg_post_id{self.id}_thres_{self.thres:.2f}_gauss_{self.gauss:.2f}_voxel_{self.voxel_cut}.mrc"
        self.output_mask = os.path.join(self.parent_path, output_mask_name)

    progress_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str, str)

    def run(self):
        if not os.path.exists(self.parent_path):
            os.mkdir(self.parent_path)

        if os.path.exists(self.input_mask) and self.skip:
            print("skip segmentation, just do post")
        else:
            self.progress_signal.emit("segmentation")
            if os.path.exists(self.input_mask):
                try:
                    os.replace(self.input_mask, self.input_mask+'~')
                except:
                    pass
            try:
                main_seg(input=self.input_raw, 
                         output=self.input_mask, 
                         model=self.input_model, 
                         batch=self.batch,
                         thread=self.threads,
                         gpuid=self.gpuid)
            except Exception as e:
                self.error_signal.emit("segmentation", repr(e))
                return

        self.progress_signal.emit("post")
        try:
            main_post(input=self.input_mask,
                      output=self.output_mask,
                      thres=self.thres,
                      gauss=self.gauss,
                      voxel_cut=self.voxel_cut)
        except Exception as e:
            self.error_signal.emit("post", repr(e))
            return


class QThread_finetune(QThread):
    def __init__(self):
        super(QThread_finetune, self).__init__()

    def setParameters(self, mUI:UI_Mpicker_FrameMemseg):
        self.input_raw = mUI.mrcPath
        self.input_mask = mUI.maskpath
        self.gpuid = mUI.gpuid_ft
        self.threads = mUI.threads_ft
        self.id = mUI.id_ft
        self.input_model = mUI.model_ft
        self.parent_path = os.path.join(mUI.parentPath, "memseg") # fix path
        self.output = os.path.join(self.parent_path, f"finetune_id{self.id}") # fix name
        self.z_range = f"{mUI.z_min},{mUI.z_max}"
        self.skip = mUI.radioButton_skip_ft.isChecked()

    progress_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str, str)

    def run(self):
        if not os.path.exists(self.parent_path):
            os.mkdir(self.parent_path)
        if not os.path.exists(self.output):
            os.mkdir(self.output)

        dataset_dir = os.path.join(self.output, "finetune_dataset") # fixed path in Mpicker_memseg
        if self.skip and os.path.isdir(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
            print("skip dataset preparation, just finetune trainning")
        else:
            self.progress_signal.emit("preparation")
            try:
                main_finetune_pre(data=self.input_raw, 
                                  mask=self.input_mask, 
                                  output=self.output, 
                                  z_range=self.z_range)
            except Exception as e:
                self.error_signal.emit("preparation", repr(e))
                return

        self.progress_signal.emit("finetune")
        try:
            main_finetune_train(data=self.input_raw, 
                                dataset=dataset_dir, 
                                model=self.input_model, 
                                gpuid=self.gpuid, 
                                thread=self.threads)
        except Exception as e:
            self.error_signal.emit("finetune", repr(e))
            return
