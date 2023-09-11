from PyQt5.QtWidgets import (QComboBox, QGroupBox, QFormLayout, QGridLayout, QApplication, QWizard, 
                             QWizardPage, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QSpinBox, 
                             QTextEdit, QRadioButton, QLabel, QLineEdit, QGroupBox, QRadioButton, 
                             QDialogButtonBox, QButtonGroup, QDoubleSpinBox)

class Param():
    def __init__(self,name, value, type_ = None, editable = True, options = None, description = None, optimizable = False, default_search_type = "choice", default_search_space = None):
        self.name = name
        self.value = value
        if type_ is not None:
            self.type_ = type_
        else:
            self.type_ = type(self.value)
        self.editable = editable
        self.options = options
        if self.options:
            assert self.value in self.options
        self.description = description
        self.has_description = description is not None

        self.optimizable = optimizable
        self.default_search_type = default_search_type
        if default_search_space:
            self.default_search_space = default_search_space
        else:
            self.default_search_space = [self.value]
    
    def get_widget(self):
        if self.type_ is bool:
            return QCheckBox(self.name)
        if self.options:
            return self.make_label(), self.make_combo_box()
        if self.type_ is float:
            return self.make_label(), self.make_spin_box(QDoubleSpinBox())
        if self.type_ is int:
            return self.make_label(), self.make_spin_box(QSpinBox())
        if self.type_ is str:
            return self.make_label(), self.make_line_edit()
    
    def make_line_edit(self):
        line_edit = QLineEdit()
        line_edit.setText(self.value)
        return line_edit
    
    def make_spin_box(self, spin_box):
        spin_box.setMaximum(1000)
        if isinstance(spin_box, QDoubleSpinBox):
            spin_box.setDecimals(5)
        spin_box.setValue(self.value)
        return spin_box
    
    def make_combo_box(self):
        combo = QComboBox()
        combo.addItems(self.options)
        combo.setCurrentIndex(self.options.index(self.value))
        return combo
    
    def make_label(self):
        return QLabel(self.name)

