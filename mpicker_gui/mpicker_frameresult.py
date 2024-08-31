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


class Mpicker_FrameResult(QFrame):
    def __init__(self, *__args):
        super(Mpicker_FrameResult, self).__init__(*__args)


    def setParameters(self, UI):
        self.UI = UI
        # FrameSetting Function
        self.UI.doubleSpinBox_resultX.valueChanged.connect(self.SpinBox_resultX)
        self.UI.doubleSpinBox_resultY.valueChanged.connect(self.SpinBox_resultY)
        self.UI.doubleSpinBox_resultZ.valueChanged.connect(self.SpinBox_resultZ)
        self.UI.spinBox_resultz.valueChanged.connect(self.SpinBox_resultz)
        self.UI.resulthorizontalSlider_z.valueChanged.connect(self.slide_resultz)
        self.UI.horizontalSlider_resultContrast.valueChanged.connect(self.slide_resultContrast)
        self.UI.horizontalSlider_resultBright.valueChanged.connect(self.slide_resultBright)
        self.UI.doubleSpinBox_resultContrast.valueChanged.connect(self.SpinBox_resultcontrast)
        self.UI.doubleSpinBox_resultBright.valueChanged.connect(self.SpinBox_resultbright)

    def slide_resultContrast(self,value):
        if self.UI.resultpixmap is not None:
            self.set_resultcontrast(round(value/10,1))

    def SpinBox_resultcontrast(self):
        if self.UI.resultpixmap is not None:
            self.set_resultcontrast(self.UI.doubleSpinBox_resultContrast.value())

    def set_resultcontrast(self,value):
        self.UI.slider_resultContrast_value  = value
        self.UI.horizontalSlider_resultContrast.setValue(self.UI.slider_resultContrast_value * 10)
        self.UI.doubleSpinBox_resultContrast.setValue(self.UI.slider_resultContrast_value)
        # self.UI.label_Press_Flag = False
        self.UI.result_tomo_min = self.UI.tomo_result_mean + (self.UI.slider_resultBright_value -
                                  self.UI.slider_resultContrast_value) * self.UI.tomo_result_std
        self.UI.result_tomo_max = self.UI.tomo_result_mean + (self.UI.slider_resultBright_value +
                                  self.UI.slider_resultContrast_value) * self.UI.tomo_result_std
        if self.UI.result_tomo_max > self.UI.result_tomo_min:
            self.UI.result_contrast = 255 / (self.UI.result_tomo_max - self.UI.result_tomo_min)
            self.UI.showResultImage()
        else:
            self.UI.result_contrast = 255
            self.UI.showResultImage()

    def slide_resultBright(self,value):
        if self.UI.resultpixmap is not None:
            self.set_resultbright(round(value/10,1))

    def SpinBox_resultbright(self):
        if self.UI.resultpixmap is not None:
            self.set_resultbright(self.UI.doubleSpinBox_resultBright.value())

    def set_resultbright(self,value):
        self.UI.slider_resultBright_value = value
        self.UI.horizontalSlider_resultBright.setValue(self.UI.slider_resultBright_value * 10)
        self.UI.doubleSpinBox_resultBright.setValue(self.UI.slider_resultBright_value)
        # self.UI.label_Press_Flag = False
        self.UI.result_tomo_min = self.UI.tomo_result_mean + (self.UI.slider_resultBright_value -
                                                              self.UI.slider_resultContrast_value) * self.UI.tomo_result_std
        self.UI.result_tomo_max = self.UI.tomo_result_mean + (self.UI.slider_resultBright_value +
                                                              self.UI.slider_resultContrast_value) * self.UI.tomo_result_std
        if self.UI.result_tomo_max > self.UI.result_tomo_min:
            self.UI.result_contrast = 255 / (self.UI.result_tomo_max - self.UI.result_tomo_min)
            self.UI.showResultImage()
        else:
            self.UI.result_contrast = 255
            self.UI.showResultImage()

    def SpinBox_resultX(self):
        if self.UI.resultpixmap is not None:
            self.UI.resultx = self.UI.doubleSpinBox_resultX.value()
            # self.UI.label_Press_Flag = False
            self.UI.showResultImage()

    def SpinBox_resultY(self):
        if self.UI.resultpixmap is not None:
            self.UI.resulty = self.UI.doubleSpinBox_resultY.value()
            # self.UI.label_Press_Flag = False
            self.UI.showResultImage()

    def SpinBox_resultZ(self):
        if self.UI.resultpixmap is not None:
            self.set_z(int(self.UI.doubleSpinBox_resultZ.value()))

    def SpinBox_resultz(self):
        if self.UI.resultpixmap is not None:
            self.set_z(self.UI.spinBox_resultz.value())

    def slide_resultz(self,value):
        if self.UI.resultpixmap is not None:
            self.set_z(value)

    def set_z(self,value):
        self.UI.resultz = value
        if self.UI.allow_showResultImage:
            self.UI.allow_showResultImage = False
            self.UI.resulthorizontalSlider_z.setValue(self.UI.resultz)
            self.UI.spinBox_resultz.setValue(self.UI.resultz)
            self.UI.doubleSpinBox_resultZ.setValue(self.UI.resultz)
            self.UI.allow_showResultImage = True
            # self.UI.label_Press_Flag = False
            self.UI.showResultImage()


