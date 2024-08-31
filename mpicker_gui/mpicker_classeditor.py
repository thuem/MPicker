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
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QLineEdit,QComboBox,QCheckBox,QRadioButton,QVBoxLayout,QButtonGroup,QListWidget,QListWidgetItem
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor, QValidator
import os,sys
from typing import List,Tuple


def color_list():
    # also use for QColor and stylesheet
    return ['blue', 'green', 'yellow', 'magenta', 'cyan', 'lime', 'orange', 'brown', 'purple']

def default_idx(i:int):
    # i start from 1
    return (i - 1) % len(color_list()) + 1

def default_name(i:int):
    # i start from 1
    return 'Class'+str(i)

def default_color(i:int):
    # i start from 1
    colors = color_list()
    idx = (i - 1) % len(colors)
    return colors[idx]


class NameValidator(QValidator):
    # no whitespace, not empty
    def __init__(self, idx:int, *args):
        super().__init__(*args)
        self.idx = idx
    def validate(self, text:str, pos:int):
        real_text = ''.join(text.split())
        if real_text == '':
            return (QValidator.Intermediate, real_text, pos)
        elif real_text == text:
            return (QValidator.Acceptable, real_text, pos)
        else:
            return (QValidator.Invalid, real_text, pos)
    def fixup(self, text:str):
        return default_name(self.idx)


class class_File(object):
    def __init__(self, file : str):
        if file is None:
            # just for convinient
            self.no_file = True
            return 
        self.no_file = False
        self.header = " ".join(["index", "class_name", "color", "show", "edit", "\n"])
        if not os.path.isfile(file):
            with open(file, mode='w') as f:
                first_row = " ".join([str(default_idx(1)), default_name(1), default_color(1), "yes", "yes", "\n"])
                f.writelines([self.header, first_row])
        self.file_path = os.path.abspath(file)
        self.read_file() # get self.data
    
    @staticmethod
    def filt_data(input_data: List[list]) -> List[Tuple[int,str,str,bool,bool]]:
        data = []
        data_final = []
        for line in input_data:
            try:
                index, name, color, show, edit = line
                index = int(index)
                if index <= 0:
                    continue
                if color not in color_list():
                    continue
                if type(show) is not bool:
                    show = True if show == "yes" else False
                if type(edit) is not bool:
                    edit = True if edit == "yes" else False
                data.append([index, name, color, show, edit])
            except:
                continue
        if len(data) == 0:
            # at least one row
            data.append([default_idx(1), default_name(1), default_color(1), True, True])

        indexs = [d[0] for d in data]
        for i in range(1, max(indexs)+1):
            # index must in order
            try:
                pos = indexs.index(i)
                data_final.append(data[pos])
            except:
                # not found, use default
                data_final.append([default_idx(i), default_name(i), default_color(i), False, False])

        edits = [d[4] for d in data_final]
        if edits.count(True) == 0:
            # must edit one
            data_final[0][3] = True
            data_final[0][4] = True
        if edits.count(True) > 1:
            pos = edits.index(True)
            for i in range(len(data_final)):
                if i == pos:
                    data_final[i][3] = True
                    data_final[i][4] = True
                else:
                    data_final[i][4] = False

        return [tuple(line) for line in data_final]

    def read_file(self):
        with open(self.file_path, mode='r') as f:
            lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        self.data = class_File.filt_data(lines)

    def update_data(self, data:List[list]):
        self.data = class_File.filt_data(data)

    def write_file(self):
        lines = []
        lines.append(self.header)
        for line in self.data:
            index, name, color, show, edit = line
            index = str(index)
            show = "yes" if show else "no"
            edit = "yes" if edit else "no"
            lines.append(" ".join([index, name, color, show, edit, "\n"]))
        with open(self.file_path, mode='w') as f:
            f.writelines(lines)

    def show_idxs(self):
        result = []
        for i in range(len(self.data)):
            if self.data[i][3] is True:
                result.append(i + 1)
        return result

    def edit_idx(self):
        for i in range(len(self.data)):
            if self.data[i][4] is True:
                return i + 1

    def idx_color(self, idx: int):
        if idx < 1:
            return
        if idx > len(self.data):
            return default_color(idx)
        color = self.data[idx-1][2]
        return color

    def idx_name(self, idx: int):
        if idx < 1:
            return
        if idx > len(self.data):
            return default_name(idx)
        name = self.data[idx-1][1]
        return name


class class_row(object):
    def __init__(self, parameters):
        super().__init__()
        self.widget_height, self.widget_width, \
        self.index_geometry, self.index_alignment, \
        self.name_geometry, self.name_alignment, \
        self.color_geometry, self.show_geometry, self.edit_geometry \
        = parameters

    def get_combo_list(self):
        #self.model = QStandardItemModel()
        self.widge_list = QListWidget()
        # for color combobox
        colors=color_list()
        for i in range(len(colors)):
            # item = QStandardItem(colors[i])
            # item.setBackground(QColor(colors[i]))
            # self.model.appendRow(item)
            item = QListWidgetItem(colors[i])
            self.widge_list.addItem(item)
            label = QLabel()
            label.setText(colors[i])
            qss = f"QLabel {{background: {colors[i]};}} QLabel:hover {{border: 2px solid black;}}"
            label.setStyleSheet(qss)
            self.widge_list.setItemWidget(item, label)
            
    def generate_widget(self, idx:int):
        # idx start from 1
        row = QWidget()
        row.resize(self.widget_height, self.widget_width)
        row.setFixedHeight(self.widget_height)
        row.setFixedWidth(self.widget_width)
        row.setObjectName("row")
        index = QLabel(row)
        index.setGeometry(self.index_geometry)
        index.setAlignment(self.index_alignment)
        index.setObjectName("index")
        name = QLineEdit(row)
        name.setGeometry(self.name_geometry)
        name.setAlignment(self.name_alignment)
        name.setObjectName("name")
        color = QComboBox(row)
        color.setGeometry(self.color_geometry)
        # color.setModel(self.model)
        self.get_combo_list() # must generate new one each time, don't know why
        color.setView(self.widge_list)
        color.setModel(self.widge_list.model())
        color.setObjectName("color")
        show = QCheckBox(row)
        show.setGeometry(self.show_geometry)
        show.setObjectName("show")
        edit = QRadioButton(row)
        edit.setGeometry(self.edit_geometry)
        edit.setObjectName("edit")

        index.setText(str(idx))
        name.setText(default_name(idx))
        name.setValidator(NameValidator(idx))
        qss = f"QComboBox{{background:{default_color(idx)}}}" # "QComboBox{background:xxx}"
        color.setCurrentIndex(default_idx(idx) - 1)
        color.setStyleSheet(qss)

        return row


class Mpicker_classeditor(QWidget):
    def __init__(self, file=None):
        super().__init__()
        # Load the ui file
        self.uifile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_edit.ui")
        uic.loadUi(self.uifile_path, self)

        parameters = [ self.widget_class1.height(), self.widget_class1.width(), 
                       self.label_index.geometry(), self.label_index.alignment(),
                       self.lineEdit_name.geometry(), self.lineEdit_name.alignment(),
                       self.comboBox_color.geometry(),
                       self.checkBox_show.geometry(),
                       self.radioButton_edit.geometry() ]
        self.row_generator = class_row(parameters)

        self.row_x = self.widget_class1.x()
        self.row_y = self.widget_class1.y()
        self.scrollAreaWidgetContents.deleteLater()

        self.vlayout = QVBoxLayout()
        self.vlayout.addStretch()
        self.vlayout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.vlayout.setContentsMargins(self.row_x, self.row_y, 0, self.row_y)
        self.scrollwidget=QWidget()
        self.scrollwidget.setLayout(self.vlayout)
        self.scrollArea.setWidget(self.scrollwidget)

        self.edit_group = QButtonGroup(self) # only one class can be edit
        self.data_file = class_File(file)

        self.pushButton_add.clicked.connect(self.add_row)
        self.pushButton_delete.clicked.connect(self.delete_row)
        self.pushButton_apply.clicked.connect(self.save_all)

    apply_signal = pyqtSignal()
        
    def set_classfile(self, file : class_File):
        self.data_file = file

    def load_file(self):
        self.clear_rows()
        for line in self.data_file.data:
            row = self.add_row()
            index:QLabel = row.findChild(QLabel, name="index")
            name:QLineEdit = row.findChild(QLineEdit, name="name")
            combo:QComboBox = row.findChild(QComboBox, name="color")
            checkbox:QCheckBox = row.findChild(QCheckBox, name="show")
            radiobutton:QRadioButton = row.findChild(QRadioButton, name="edit")
            index.setText(str(line[0])) # in fact not necessary
            name.setText(line[1])
            combo.setCurrentText(line[2])
            checkbox.setChecked(line[3])
            radiobutton.setChecked(line[4])

    def add_row(self):
        idx = self.vlayout.count() # always has one stretch
        row = self.row_generator.generate_widget(idx)
        combo:QComboBox = row.findChild(QComboBox, name="color")
        combo.currentIndexChanged.connect(self.color_change)
        radiobutton:QRadioButton = row.findChild(QRadioButton, name="edit")
        radiobutton.toggled.connect(self.edit_click)
        self.edit_group.addButton(radiobutton)
        if idx==1: # the first row
            radiobutton.setChecked(True)
        self.vlayout.insertWidget(idx-1, row)
        return row

    def delete_row(self):
        idx = self.vlayout.count()
        if idx <= 2: # only one row and one stretch
            return
        else:
            row = self.vlayout.itemAt(idx-2).widget()
            radiobutton:QRadioButton = row.findChild(QRadioButton, name="edit")
            if radiobutton.isChecked():
                # always make one checked
                row2 = self.vlayout.itemAt(idx-3).widget()
                radiobutton2:QRadioButton = row2.findChild(QRadioButton, name="edit")
                radiobutton2.setChecked(True)
            self.vlayout.removeWidget(row)
            row.deleteLater()

    def clear_rows(self):
        number = self.vlayout.count()
        for i in range(number - 2, -1, -1):
            row = self.vlayout.itemAt(i).widget()
            row.deleteLater()

    def color_change(self, idx):
        combo:QComboBox = self.sender()
        color = color_list()[idx]
        combo.setStyleSheet(f"QComboBox{{background:{color}}}")

    def edit_click(self, checked):
        radiobutton:QRadioButton = self.sender()
        checkbox:QCheckBox = radiobutton.parent().findChild(QCheckBox, name="show")
        if checked:
            checkbox.setChecked(True)
            checkbox.setEnabled(False)
        else:
            checkbox.setEnabled(True)

    def save_all(self):
        data = []
        row_list:List[QWidget] = self.scrollwidget.findChildren(QWidget, name="row")
        for row in row_list:
            index:QLabel = row.findChild(QLabel, name="index")
            name:QLineEdit = row.findChild(QLineEdit, name="name")
            combo:QComboBox = row.findChild(QComboBox, name="color")
            checkbox:QCheckBox = row.findChild(QCheckBox, name="show")
            radiobutton:QRadioButton = row.findChild(QRadioButton, name="edit")
            line = [index.text(), name.text(), combo.currentText(), checkbox.isChecked(), radiobutton.isChecked()]
            data.append(line)
        if self.data_file.no_file:
            print("not save, no file")
            print(class_File.filt_data(data))
        else:
            self.data_file.update_data(data)
            self.data_file.write_file()
            self.apply_signal.emit()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Form = Mpicker_classeditor("tmpclass.txt")
    # Form.set_file_path("tmpclass.txt")
    Form.load_file()
    Form.show()
    sys.exit(app.exec_())