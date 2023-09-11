from PyQt5.QtWidgets import (QComboBox, QGroupBox, QFormLayout, QGridLayout, QApplication, QWizard, 
                             QWizardPage, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QSpinBox, 
                             QTextEdit, QRadioButton, QLabel, QLineEdit, QGroupBox, QRadioButton, 
                             QDialogButtonBox, QButtonGroup)
from scipy.io import loadmat
from PyQt5 import QtCore, QtGui
from utils import *
from model import BrainGB, BrainGNN

class FilePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Select an flist file")
        self.setSubTitle("Choose a textfile containing a list of .mat or .npy binary or weighted matrices")
        self.setLayout(QGridLayout())

        self.openFileButton = QPushButton("Open Flist File")
        self.layout().addWidget(self.openFileButton, 0,0,1,4)
        self.fileLabel = QLabel()
        self.layout().addWidget(self.fileLabel, 0,2,1,4)


        self.augmentedCheckbox = QCheckBox("Aug Data      |")
        self.layout().addWidget(self.augmentedCheckbox, 1,0,1,2)

        spinBoxLabel = QLabel("Aug Factor")
        self.augmentationFactor = QSpinBox()
        self.augmentationFactor.setDisabled(True)
        self.layout().addWidget(spinBoxLabel,1,2,1,1)
        self.layout().addWidget(self.augmentationFactor, 1,3,1,1)

        chooseLabel = QLabel("Choose variable key")
        self.labelChoose = QComboBox()
        self.labelChoose.setEnabled(False)
        self.labelChoose.currentIndexChanged.connect(self.labelChooseChange)
        self.layout().addWidget(chooseLabel, 2,0,1,2)
        self.layout().addWidget(self.labelChoose,2,2,1,2)

        spinBoxLabel = QLabel("Threshold percentile")
        self.thresholdLevel = QSpinBox()
        self.thresholdLevel.setEnabled(False)
        self.layout().addWidget(spinBoxLabel, 3,0,1,2)
        self.layout().addWidget(self.thresholdLevel, 3,2,1,1)

        self.textBox = QLabel()
        self.layout().addWidget(self.textBox,2,0,1,4)

        self.openFileButton.clicked.connect(self.openFileDialog)
        self.augmentedCheckbox.toggled.connect(self.augmentationFactor.setEnabled)

        self.extensions = ["mat", "npy"]

    def openFileDialog(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)", options=options)
        self.filepath = filePath
        self.fileLabel.setText(self.filepath)
        self.getFlistAttributes()
        self.textBox.clear()
        self.textBox.setText(f"num_examples: {len(self.files)}")
        
    def getFlistAttributes(self, file):
        with open(self.filepath, "r") as f:
            self.files = [file.strip() for file in f.readlines()]
        self.extension = self.files[0].split(".")[-1]
        self.matKeys = loadmat(self.files[0]).keys()
        self.labelChoose.clear()
        self.labelChoose.addItems(self.matKeys)

    def augmentationCheckBoxClicked(self):
        if self.augmentedCheckbox.isChecked():
            self.augmentationFactor.setEnabled(True)
            self.augmentationFactor.setValue(15)
        else:
            self.augmentationFactor.setEnabled(False)
            self.augmentationFactor.setValue(True)

    def labelChooseChange(self):
        key = self.labelChoose.currentText()
        sample = loadmat(self.files[0])[key]
        self.shape = sample.shape
        self.dtype = sample.dtype
        self.is_binary = is_binary(sample)
        if self.is_binary:
            self.thresholdLevel.setEnabled(True)
            self.thresholdLevel.setValue(10)
        else:
            self.thresholdLevel.setEnabled(False)
    
    def addDataAttributesToText(self):
        self.textBox.setText(f"{self.textBox.text()}\nshape: {self.shape}\ndtype: {self.dtype}\nis_binary: {self.is_binary}")


class ModelPage(QWizardPage):
    def __init__(self):
        super().__init__()

        self.setTitle("Build your GNN")
        self.setSubTitle("Select from preimplemented models or customize componentry")

        myFont=QtGui.QFont()
        myFont.setBold(True)

        self.setWindowTitle("Graph Neural Network Customization")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()
        
        boldLabel1 = QLabel("Use Preimplemented Model")
        boldLabel1.setFont(myFont)
        self.layout.addWidget(boldLabel1)
        self.models_bg = QButtonGroup()
        self.models_bg.setExclusive(False)
        self.use_brain_cnn = QCheckBox("BrainNetCNN")
        self.use_brain_gnn = QCheckBox("BrainGNN")
        self.models_bg.addButton(self.use_brain_cnn,1)
        self.models_bg.addButton(self.use_brain_gnn,2)
        self.models_bg.buttonClicked.connect(self.use_preimpl_model)
        self.layout.addWidget(self.use_brain_cnn)
        self.layout.addWidget(self.use_brain_gnn)

        self.groupboxes = []

        label = QLabel("Or Customize GNN")
        label.setFont(myFont)
        self.layout.addWidget(label)

        self.node_features_group = QGroupBox("Node Features")
        self.node_features_layout = QFormLayout()
        self.node_features_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.node_features_group.setLayout(self.node_features_layout)
        self.groupboxes.append(self.node_features_group)

        self.graph_conv_group = QGroupBox("Graph Convolution Layer Type")
        self.graph_conv_attention_checkbox = QCheckBox("USE ATTENTION")
        self.graph_conv_layout = QFormLayout()
        self.graph_conv_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.graph_conv_group.setLayout(self.graph_conv_layout)
        self.groupboxes.append(self.graph_conv_group)

        self.pooling_group = QGroupBox("Pooling Strategies")
        self.pooling_layout = QFormLayout()
        self.pooling_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.pooling_group.setLayout(self.pooling_layout)
        self.groupboxes.append(self.pooling_group)

        self.layout.addWidget(self.node_features_group)
        self.layout.addWidget(self.graph_conv_group)
        self.layout.addWidget(self.pooling_group)

        self.setLayout(self.layout)

        # Node Features
        self.node_features_options = ["identity", "eigen", "degree", "degree profile", "connection profile"]
        self.node_features_radios = []
        for option in self.node_features_options:
            radio = QRadioButton(option)
            self.node_features_layout.addWidget(radio, )
            self.node_features_radios.append(radio)
        self.node_features_radios[0].setChecked(True)

        # Graph Convolution Layer Type
        self.graph_conv_options_mp = ["edge weighted", "bin concat", "edge weight concat", "node edge concat", "node concat"]
        self.graph_conv_options_ma = ["attention weighted", "edge weighted attention", "attention edge sum", "node edge concat with attention", "node concat w attention"]
        self.graph_conv_radios = []

        self.graph_conv_layout.addWidget(self.graph_conv_attention_checkbox)

        for option in self.graph_conv_options_mp:
            radio = QRadioButton(option)
            self.graph_conv_layout.addRow(radio)
            self.graph_conv_radios.append(radio)
        self.graph_conv_radios[0].setChecked(True)

        # Pooling Strategies
        self.pooling_options = ["mean pooling", "sum pooling", "concat pooling", "diffpool"]
        self.pooling_radios = []
        for option in self.pooling_options:
            radio = QRadioButton(option)
            self.pooling_layout.addWidget(radio)
            self.pooling_radios.append(radio)
        self.pooling_radios[0].setChecked(True)


        self.graph_conv_attention_checkbox.stateChanged.connect(self.update_graph_conv_options)
    
    def use_preimpl_model(self):
        self.toggle_gbs(self.models_bg.checkedButton()== None)
        if self.models_bg.checkedId() ==2:
            self.use_brain_cnn.setChecked(False)
        elif self.models_bg.checkedId()==1:
            self.use_brain_gnn.setChecked(False)

    def toggle_gbs(self, b):
        for groupbox in self.groupboxes:
                groupbox.setEnabled(b)
        

    def update_graph_conv_options(self):
        use_attention = self.graph_conv_attention_checkbox.isChecked()
        options_to_use = self.graph_conv_options_ma if use_attention else self.graph_conv_options_mp

        for radio, new_text in zip(self.graph_conv_radios, options_to_use):
            radio.setText(new_text)

    def get_data(self):
        data = {
            "Node Features": [radio.text() for radio in self.node_features_radios if radio.isChecked()],
            "message-passing": [],
            "message-passing with attention": [],
            "Pooling Strategies": [radio.text() for radio in self.pooling_radios if radio.isChecked()]
        }
    
        if self.graph_conv_attention_checkbox.isChecked():
            mp_key = "message-passing with attention"
        else:
            mp_key = "message-passing"

        data[mp_key] = [radio.text() for radio in self.graph_conv_radios if radio.isChecked()]

        return data


class HyperParamDialog(QWizardPage):
    def __init__(self):
        super().__init__()

        self.setTitle("Select Hyperparameters")
        self.setSubTitle("Choose hyperparameters, or define parameter search spaces")

        self.bold_font=QtGui.QFont()
        self.bold_font.setBold(True)

        self.setWindowTitle("Select Parameters")
        self.make_layout()

    def make_layout(self):
        self.layout = QGridLayout()
        self.setLayout(self.layout)
   
    def initializePage(self):
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)

        modelType = self.wizard().page(1).use_brain_gnn.isChecked() or self.wizard().page(1).use_brain_cnn.isChecked()
        if modelType:
            model = BrainGNN
        else:
            model = BrainGB
        
        self.params = []
        
        self.row = -1
        for key, value in model.params.items():
            self.row+=1
            self.layout.addWidget(self.make_bold_label(key),self.row,0,1,2)
            for param in value:
                print(param.name)
                self.params.append(param)
                self.row+=1
                widget = param.get_widget()
                if type(widget) is tuple:
                    label, widget = widget
                    self.layout.addWidget(label,self.row,0,1,2)
                    self.layout.addWidget(widget, self.row, 2,1,2)
                else:
                    self.layout.addWidget(widget, self.row, 0,1,2)
        self.configure_nni_dropdown()

    def configure_nni_dropdown(self):
        self.row+=1
        self.nni_dropdown = QComboBox()
        self.nni_dropdown.addItems(["None", "Random", "GridSearch", "TPE", "Evolution", "Anneal", "Evolution", 
                                    "Hyperband", "SMAC", "Batch", "Hyperband", "Metis", "BOHB", "GP", "PBT", "DNGO"])
        self.layout.addWidget(QLabel("Hyperparameter Optimization Algorithm"), self.row,0,1,2)
        self.layout.addWidget(self.nni_dropdown, self.row, 2,1,2)
        self.nni_dropdown.currentIndexChanged.connect(self.nni_dropdown_change)
        self.row+=1

        self.num_trials_spin = QSpinBox()
        self.num_trials_spin.setValue(10)
        self.num_trials_spin.setMaximum(1000)
        self.num_trials_spin.setEnabled(False)
        self.layout.addWidget(QLabel("max trails"), self.row, 0,1,2)
        self.layout.addWidget(self.num_trials_spin, self.row,2,1,2)
        self.row+=1

        self.max_time_edit = QLineEdit()
        self.max_time_edit.setText("24hr")
        self.max_time_edit.setEnabled(False)
        self.layout.addWidget(QLabel("max time"), self.row, 0,1,2)
        self.layout.addWidget(self.max_time_edit, self.row,2,1,2)
        self.row+=1
    
    def nni_dropdown_change(self):
        if self.nni_dropdown.currentText != None:
            self.add_text_edit()
            self.make_search_space_json()
            self.num_trials_spin.setEnabled(True)
            self.max_time_edit.setEnabled(True)
        else:
            self.layout.removeWidget(self.search_space_text)
            self.num_trials_spin.setEnabled(False)
            self.max_time_edit.setEnabled(False)

    def add_text_edit(self):
        self.search_space_text = QTextEdit()
        self.layout.addWidget(self.search_space_text, self.row, 0, 8,4)
    
    def make_search_space_json(self):
        self.search_space = {}
        for param in self.params:
            if param.optimizable:
                self.add_param_to_search_space(param)
        self.search_space_text.clear()
        self.search_space_text.setText(custom_json_dump(self.search_space))


    def add_param_to_search_space(self, param):
        name = param.name
        search_type = param.default_search_type
        space  = param.default_search_space
        self.search_space[name] = {"_type": search_type, "_value": space}
    
    def make_bold_label(self, text):
        label = QLabel(text)
        label.setFont(self.bold_font)
        return label