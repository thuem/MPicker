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

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QValidator, QIntValidator
import os, sys
import configparser
import Mpicker_epicker
for_fake_import = False
if for_fake_import:
    import Mpicker_gui
# from Ui_epicker import Ui_Mpicker_FrameMemseg


# Define function to import external files when using PyInstaller. ??
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class SpaceValidator(QValidator):
    # no whitespace
    def __init__(self, *args):
        super().__init__(*args)
    def validate(self, text:str, pos:int):
        real_text = ''.join(text.lstrip('-').split())
        if real_text == text:
            return (QValidator.Acceptable, real_text, pos)
        else:
            return (QValidator.Invalid, real_text, pos)


class Mpicker_Epickergui(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the ui file
        self.uifile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epicker.ui")
        uic.loadUi(self.uifile_path, self)
        self.lineEdit_zrange_pick.setValidator(SpaceValidator())
        self.lineEdit_gpuid_pick.setValidator(SpaceValidator())
        self.lineEdit_outid.setValidator(SpaceValidator())
        self.lineEdit_setid.setValidator(SpaceValidator())
        self.lineEdit_classrange.setValidator(SpaceValidator())
        self.lineEdit_zrange_train.setValidator(SpaceValidator())
        self.lineEdit_gpuid_train.setValidator(SpaceValidator())
        self.lineEdit_edgex.setValidator(QIntValidator(0, 9999))
        self.lineEdit_edgey.setValidator(QIntValidator(0, 9999))

        self.checkBox_model.toggled.connect(self.fromModel_toggle)
        self.checkBox_exemplar.toggled.connect(self.exemplar_toggle)
        self.checkBox_trainset.toggled.connect(self.trainset_toggle)
        self.Button_pick.clicked.connect(self.run_pick)
        self.Button_generate.clicked.connect(self.run_generate)
        self.Button_train.clicked.connect(self.run_train)
        self.Button_rmtmp.clicked.connect(self.remove_tmpfile)
        self.Button_selectModel_pick.clicked.connect(self.selectModel_pick)
        self.Button_selectModel_train.clicked.connect(self.selectModel_train)
        self.Button_selectExemplar.clicked.connect(self.selectExemplar)
        self.Button_selectTrainset.clicked.connect(self.selectTrainset)

    def setParameters(self, UI):
        self.UI = UI
        if self.check_UI_OK():
            self.get_in_path()
            self.read_config()

    def check_UI_OK(self):
        if self.UI.pickfolder is not None and self.UI.MRC_path is not None \
            and self.UI.showResult_path is not None and self.UI.savetxt_path is not None:
            return True
        else:
            return False

    def get_in_path(self):
        # update info about mainGUI
        self.volume_path = self.UI.showResult_path
        self.coord_path = self.UI.savetxt_path
        self.tomo_name = os.path.splitext(os.path.basename(self.UI.ini_config_path))[0]
        self.base_path = os.path.join(self.UI.pickfolder, self.tomo_name)
        self.config_path = os.path.join(self.base_path, self.UI.epicker_config_name)
        self.epicker_path = os.path.join(self.base_path, "epicker")
        if not os.path.isdir(self.epicker_path):
            os.mkdir(self.epicker_path)

    def get_out_path(self):
        self.pickout_path = self.volume_path.replace("_result.mrc", f"_epickerCoord_id{self.config['pick']['outid']}.txt")
        self.tmpfile_path = self.volume_path.replace("_result.mrc", f"_epickerTmp{self.config['pick']['number']}_id{self.config['pick']['outid']}.txt")
        self.trainset_dir = os.path.join(self.epicker_path, f"trainset_id{self.config['train']['setid']}")
        trainout_name = f"trainset_id{self.config['train']['setid']}_bs{self.config['train']['batch']}_result"
        self.trainout_dir = os.path.join(self.epicker_path, trainout_name)
        if self.config.getboolean('train', 'trainset'):
            # user specified path
            self.trainset_dir_train = self.config['train']['trainsetDir']
        else:
            # default path
            self.trainset_dir_train = self.trainset_dir

    def update_config(self):
        # update info about epickerGUI
        config = configparser.ConfigParser()
        config['pick'] = {}
        config['train'] = {}
        config['other'] = {}
        config.set('pick', 'epicker', self.lineEdit_epicker_pick.text())
        config.set('pick', 'edgex', self.lineEdit_edgex.text())
        config.set('pick', 'edgey', self.lineEdit_edgey.text())
        config.set('pick', 'thres', f'{self.doubleSpinBox_thres.value() :.3f}')
        config.set('pick', 'number', str(self.spinBox_number.value()))
        config.set('pick', 'distance', f'{self.doubleSpinBox_distance.value() :.1f}')
        config.set('pick', 'zrange', self.lineEdit_zrange_pick.text())
        config.set('pick', 'gpuid', self.lineEdit_gpuid_pick.text())
        config.set('pick', 'outid', self.lineEdit_outid.text())
        config.set('pick', 'InputModel', self.Text_InputModel_pick.toPlainText())
        config.set('pick', 'load2gui', str(self.checkBox_load2gui.isChecked()))
        config.set('pick', 'tmpfile', str(self.checkBox_tmpfile.isChecked()))
        config.set('train', 'setid', self.lineEdit_setid.text())
        config.set('train', 'sparse', str(self.checkBox_sparse.isChecked()))
        config.set('train', 'classrange', self.lineEdit_classrange.text())
        config.set('train', 'zrange', self.lineEdit_zrange_train.text())
        config.set('train', 'epicker', self.lineEdit_epicker_train.text())
        config.set('train', 'gpuid', self.lineEdit_gpuid_train.text())
        config.set('train', 'fromModel', str(self.checkBox_model.isChecked()))
        config.set('train', 'InputModel', self.Text_InputModel_train.toPlainText())
        config.set('train', 'exemplar', str(self.checkBox_exemplar.isChecked()))
        config.set('train', 'exemplarDir', self.Text_exemplar.toPlainText())
        config.set('train', 'exemplarNum', str(self.spinBox_exemplarnum.value()))
        config.set('train', 'trainset', str(self.checkBox_trainset.isChecked()))
        config.set('train', 'trainsetDir', self.Text_trainset.toPlainText())
        config.set('train', 'epoch', str(self.spinBox_epoch.value()))
        config.set('train', 'batch', str(self.spinBox_batch.value()))
        config.set('other', 'padsize', str(self.spinBox_padsize.value()))
        self.config = config

    def write_config(self):
        try:
            with open(self.config_path, 'w') as f:
                self.config.write(f)
        except:
            print("fail to write epicker config")

    def read_config(self):
        if not os.path.isfile(self.config_path):
            return
        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
            # will not use fallback in normal case
            self.lineEdit_epicker_pick.setText(config.get('pick', 'epicker', fallback="epicker.sh"))
            self.lineEdit_edgex.setText(config.get('pick', 'edgex', fallback="5"))
            self.lineEdit_edgey.setText(config.get('pick', 'edgey', fallback="10"))
            self.doubleSpinBox_thres.setValue(config.getfloat('pick', 'thres', fallback=0.1))
            self.spinBox_number.setValue(config.getint('pick', 'number', fallback=500))
            self.doubleSpinBox_distance.setValue(config.getfloat('pick', 'distance', fallback=5))
            self.lineEdit_zrange_pick.setText(config.get('pick', 'zrange', fallback=""))
            self.lineEdit_gpuid_pick.setText(config.get('pick', 'gpuid', fallback="0"))
            self.lineEdit_outid.setText(config.get('pick', 'outid', fallback="0"))
            self.Text_InputModel_pick.setPlainText(config.get('pick', 'InputModel', fallback=""))
            self.checkBox_load2gui.setChecked(config.getboolean('pick', 'load2gui', fallback=True))
            self.checkBox_tmpfile.setChecked(config.getboolean('pick', 'tmpfile', fallback=True))
            self.lineEdit_setid.setText(config.get('train', 'setid', fallback="0"))
            self.checkBox_sparse.setChecked(config.getboolean('train', 'sparse', fallback=False))
            self.lineEdit_classrange.setText(config.get('train', 'classrange', fallback=""))
            self.lineEdit_zrange_train.setText(config.get('train', 'zrange', fallback=""))
            self.lineEdit_epicker_train.setText(config.get('train', 'epicker', fallback="epicker_train.sh"))
            self.lineEdit_gpuid_train.setText(config.get('train', 'gpuid', fallback=""))
            self.checkBox_model.setChecked(config.getboolean('train', 'fromModel', fallback=False))
            self.Text_InputModel_train.setPlainText(config.get('train', 'InputModel', fallback=""))
            self.checkBox_exemplar.setChecked(config.getboolean('train', 'exemplar', fallback=False))
            self.Text_exemplar.setPlainText(config.get('train', 'exemplarDir', fallback=""))
            self.spinBox_exemplarnum.setValue(config.getint('train', 'exemplarNum', fallback=0))
            self.checkBox_trainset.setChecked(config.getboolean('train', 'trainset', fallback=False))
            self.Text_trainset.setPlainText(config.get('train', 'trainsetDir', fallback=""))
            self.spinBox_epoch.setValue(config.getint('train', 'epoch', fallback=120))
            self.spinBox_batch.setValue(config.getint('train', 'batch', fallback=4))
            self.spinBox_padsize.setValue(config.getint('other', 'padsize', fallback=-1))
        except:
            print("fail to read epicker config")

    def fromModel_toggle(self, checked):
        if checked:
            self.Text_InputModel_train.setEnabled(True)
            self.Button_selectModel_train.setEnabled(True)
        else:
            self.Text_InputModel_train.setEnabled(False)
            self.Button_selectModel_train.setEnabled(False)

    def exemplar_toggle(self, checked):
        if checked:
            self.Text_exemplar.setEnabled(True)
            self.Button_selectExemplar.setEnabled(True)
        else:
            self.Text_exemplar.setEnabled(False)
            self.Button_selectExemplar.setEnabled(False)

    def trainset_toggle(self, checked):
        if checked:
            self.Text_trainset.setEnabled(True)
            self.Button_selectTrainset.setEnabled(True)
        else:
            self.Text_trainset.setEnabled(False)
            self.Button_selectTrainset.setEnabled(False)

    def selectModel_pick(self):
        if self.check_UI_OK():
            self.get_in_path()
        else:
            return
        start_path = os.path.dirname(self.Text_InputModel_pick.toPlainText())
        if not os.path.isdir(start_path):
            start_path = self.epicker_path
        fname = QFileDialog.getOpenFileName(self, "Open Model File", start_path, "PTH Files (*.pth);;All Files (*)")[0]
        if fname != '':
            self.Text_InputModel_pick.setText(fname)

    def remove_tmpfile(self):
        if not self.check_UI_OK():
            return
        self.get_in_path()
        self.update_config()
        self.write_config()
        self.get_out_path()
        if os.path.isfile(self.tmpfile_path):
            os.remove(self.tmpfile_path)
            print("remove", self.tmpfile_path)

    def selectModel_train(self):
        if self.check_UI_OK():
            self.get_in_path()
        else:
            return
        start_path = os.path.dirname(self.Text_InputModel_train.toPlainText())
        if not os.path.isdir(start_path):
            start_path = self.epicker_path
        fname = QFileDialog.getOpenFileName(self, "Open Model File", start_path, "PTH Files (*.pth);;All Files (*)")[0]
        if fname != '':
            self.Text_InputModel_train.setText(fname)

    def selectExemplar(self):
        if self.check_UI_OK():
            self.get_in_path()
        else:
            return
        start_path = os.path.dirname(self.Text_exemplar.toPlainText())
        if not os.path.isdir(start_path):
            start_path = self.epicker_path
        fname = QFileDialog.getExistingDirectory(self, "Select exemplar Folder", start_path)
        if fname != '':
            self.Text_exemplar.setText(fname)

    def selectTrainset(self):
        if self.check_UI_OK():
            self.get_in_path()
        else:
            return
        start_path = os.path.dirname(self.Text_trainset.toPlainText())
        if not os.path.isdir(start_path):
            start_path = self.epicker_path
        fname = QFileDialog.getExistingDirectory(self, "Select trainset Folder", start_path)
        if fname != '':
            self.Text_trainset.setText(fname)

    def show_error(self, err):
        QMessageBox.warning(self, "Error", err, QMessageBox.Ok)

    def run_pick(self):
        if not self.check_UI_OK():
            return
        self.get_in_path()
        self.update_config()
        self.write_config()
        self.get_out_path()

        if not os.path.isfile(self.config['pick']['InputModel']):
            QMessageBox.warning(self, "Input Error", "You must provide a model file", QMessageBox.Ok)
            return

        self.pick_success = False
        self.pick_volume = self.volume_path
        def success():
            self.pick_success = True
        self.load2gui = self.config.getboolean('pick', 'load2gui')
        self.tmpfile = self.config.getboolean('pick', 'tmpfile')

        self.Text_progress.setTextColor(Qt.red)
        self.Text_progress.setText("Picking...")
        self.Button_pick.setEnabled(False)
        self.Thread_pick = QThread_pick()
        self.Thread_pick.setParameters(self)
        self.Thread_pick.finished.connect(self.pick_finish)
        self.Thread_pick.error_signal.connect(self.show_error)
        self.Thread_pick.success_signal.connect(success)
        self.Thread_pick.start()

    def pick_finish(self):
        self.Text_progress.setTextColor(Qt.black)
        self.Text_progress.setText("No Progress Running")
        self.Button_pick.setEnabled(True)
        if self.pick_success and self.load2gui and self.check_UI_OK() and self.volume_path == self.UI.showResult_path:
            if os.path.isfile(self.pickout_path):
                self.UI.graphicsView_result.load_coord_file(self.pickout_path)

    def run_generate(self):
        if not self.check_UI_OK():
            return
        self.get_in_path()
        self.update_config()
        self.write_config()
        self.get_out_path()

        if os.path.isfile(self.coord_path) and os.path.getsize(self.coord_path):
            pass
        else:
            QMessageBox.warning(self, "Input Error", "Save some manual picked particles at first", QMessageBox.Ok)
            return

        self.Text_progress.setTextColor(Qt.red)
        self.Text_progress.setText("Generating trainset...")
        self.Button_generate.setEnabled(False)
        self.Thread_generate = QThread_generate()
        self.Thread_generate.setParameters(self)
        self.Thread_generate.finished.connect(self.generate_finish)
        self.Thread_generate.error_signal.connect(self.show_error)
        self.Thread_generate.start()

    def generate_finish(self):
        self.Text_progress.setTextColor(Qt.black)
        self.Text_progress.setText("No Progress Running")
        self.Button_generate.setEnabled(True)

    def run_train(self):
        if not self.check_UI_OK():
            return
        self.get_in_path()
        self.update_config()
        self.write_config()
        self.get_out_path()

        if not os.path.isdir(self.trainset_dir_train):
            QMessageBox.warning(self, "Input Error", "Generate the trainset at first", QMessageBox.Ok)
            return
        if self.config.getboolean('train', 'exemplar') and not self.config.getboolean('train', 'fromModel'):
            QMessageBox.warning(self, "Input Error", 
                                "Require both model and exemplar for continual training", 
                                QMessageBox.Ok)
            return

        self.Text_progress.setTextColor(Qt.red)
        self.Text_progress.setText("Training...")
        self.Button_train.setEnabled(False)
        self.Thread_train = QThread_train()
        self.Thread_train.setParameters(self)
        self.Thread_train.finished.connect(self.train_finish)
        self.Thread_train.error_signal.connect(self.show_error)
        self.Thread_train.start()

    def train_finish(self):
        self.Text_progress.setTextColor(Qt.black)
        self.Text_progress.setText("No Progress Running")
        self.Button_train.setEnabled(True)

    def closeEvent(self,event):
        if self.Text_progress.toPlainText() != "No Progress Running":
            reply = QMessageBox.question(self, 
                                        'Progress is running',
                                        "The Progress may not stop, do you want to exit?",
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()



class QThread_pick(QThread):
    def __init__(self):
        super().__init__()

    def setParameters(self, gui:Mpicker_Epickergui):
        self.model = gui.config['pick']['InputModel']
        self.volume = gui.volume_path
        self.z_range = gui.config['pick']['zrange']
        self.pick_out = gui.pickout_path
        self.tmpfile = gui.tmpfile
        self.tmpfile_path = gui.tmpfile_path
        self.thres = gui.config['pick']['thres']
        self.max_num = gui.config['pick']['number']
        self.dist = gui.config['pick']['distance']
        self.edgex = gui.config['pick']['edgex']
        self.edgey = gui.config['pick']['edgey']
        self.pad = gui.config['other']['padsize']
        self.gpuid = gui.config['pick']['gpuid']
        self.epicker_path = gui.config['pick']['epicker']

    error_signal = pyqtSignal(str)
    success_signal = pyqtSignal()

    def run(self):
        cmd = f'{sys.executable} {Mpicker_epicker.__file__} --mode pick --model {self.model} --volume {self.volume} ' \
            + f'--pick_out {self.pick_out} --thres {self.thres} --max_num {self.max_num} --sigma 2.0 ' \
            + f'--edgex {self.edgex} --edgey {self.edgey} --gpuid {self.gpuid} --epicker_path "{self.epicker_path}" '
        if self.z_range != '':
            cmd += f'--z_range {self.z_range} '
        if float(self.dist) >= 0:
            cmd += f'--dist {self.dist} '
        if int(self.pad)  > 0:
            cmd += f'--pad {self.pad} '
        if self.tmpfile:
            if os.path.isfile(self.tmpfile_path):
                cmd += f'--tmp_in {self.tmpfile_path} '
            else:
                cmd += f'--tmp_out {self.tmpfile_path} '

        print(f'\033[0;35m{cmd}\033[0m')
        s = os.system(cmd)
        if s != 0:
            print("exit code", s)
            self.error_signal.emit("picking failed, see terminal for detail")
            return
        else:
            self.success_signal.emit()


class QThread_generate(QThread):
    def __init__(self):
        super().__init__()

    def setParameters(self, gui:Mpicker_Epickergui):
        self.volume = gui.volume_path
        self.label_dir = gui.trainset_dir
        self.label_coord = gui.coord_path
        self.label_pre = gui.tomo_name.replace(' ', '') + '_'
        self.z_range = gui.config['train']['zrange']
        self.filt_range = gui.config['train']['classrange']
        self.pad = gui.config['other']['padsize']

    error_signal = pyqtSignal(str)

    def run(self):
        cmd = f'{sys.executable} {Mpicker_epicker.__file__} --mode label --volume {self.volume} ' \
            + f'--label_dir {self.label_dir} --label_coord {self.label_coord} --label_pre {self.label_pre} '
        if self.z_range != '':
            cmd += f'--z_range {self.z_range} '
        if self.filt_range != '':
            cmd += f'--filt_range {self.filt_range} --filt_col 17 '
        if int(self.pad)  > 0:
            cmd += f'--pad {self.pad} '

        print(f'\033[0;35m{cmd}\033[0m')
        s = os.system(cmd)
        if s != 0:
            print("exit code", s)
            self.error_signal.emit("generating label failed, see terminal for detail")
            return
        

class QThread_train(QThread):
    def __init__(self):
        super().__init__()

    def setParameters(self, gui:Mpicker_Epickergui):
        self.label_dir = gui.trainset_dir_train
        self.train_out = gui.trainout_dir
        self.gpuid = gui.config['train']['gpuid']
        self.epicker_path = gui.config['train']['epicker']
        self.sparse = gui.config.getboolean('train', 'sparse')
        self.fromModel = gui.config.getboolean('train', 'fromModel')
        self.model = gui.config['train']['InputModel']
        self.exemplar = gui.config.getboolean('train', 'exemplar')
        self.exemplarDir = gui.config['train']['exemplarDir']
        self.exemplarNum = gui.config['train']['exemplarNum']
        self.epoch = gui.config['train']['epoch']
        self.batchsize = gui.config['train']['batch']

    error_signal = pyqtSignal(str)

    def run(self):
        cmd = f'{sys.executable} {Mpicker_epicker.__file__} --mode train --label_dir {self.label_dir} --train_out {self.train_out} --gpuid {self.gpuid} ' \
            + f'--epicker_path "{self.epicker_path}" --batchsize {self.batchsize} --epoch {self.epoch} --exemplar_num {self.exemplarNum} '
        if self.sparse:
            cmd += f'--sparse '
        if self.fromModel and os.path.isfile(self.model):
            cmd += f'--model {self.model} '
        if self.exemplar and os.path.isdir(self.exemplarDir):
            cmd += f'--exemplar_dir {self.exemplarDir} '

        print(f'\033[0;35m{cmd}\033[0m')
        s = os.system(cmd)
        if s != 0:
            print("exit code", s)
            self.error_signal.emit("training failed, see terminal for detail")
            return