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

from PyQt5 import QtCore, QtWidgets

class MySplitter(QtWidgets.QSplitter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.press_flag=False
    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        self.press_flag=False
        return super().leaveEvent(a0) 
    def enterEvent(self, a0: QtCore.QEvent) -> None:
        self.press_flag=True
        return super().enterEvent(a0)