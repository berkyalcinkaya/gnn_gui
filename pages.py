from PyQt5.QtWidgets import (QComboBox, QGroupBox, QFormLayout, QGridLayout, QApplication, QWizard, 
                             QWizardPage, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QSpinBox, 
                             QTextEdit, QRadioButton, QLabel, QLineEdit, QGroupBox, QRadioButton, 
                             QDialogButtonBox, QButtonGroup)
from scipy.io import loadmat
from PyQt5 import QtCore, QtGui

class FilePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Build your graph neural network")
        self.setSubTitle("Select an flist file")
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
        self.layout().addWidget(chooseLabel, 2,0,1,2)
        self.layout().addWidget(self.labelChoose,2,2,1,2)

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
        self.write

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



class ModelPage(QWizardPage):
    def __init__(self):
        super().__init__()

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
        self.node_features_group.setLayout(self.node_features_layout)
        self.groupboxes.append(self.node_features_group)

        self.graph_conv_group = QGroupBox("Graph Convolution Layer Type")
        self.graph_conv_attention_checkbox = QCheckBox("USE ATTENTION")
        self.graph_conv_layout = QFormLayout()
        self.graph_conv_group.setLayout(self.graph_conv_layout)
        self.groupboxes.append(self.graph_conv_group)

        self.pooling_group = QGroupBox("Pooling Strategies")
        self.pooling_layout = QFormLayout()
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
            self.node_features_layout.addWidget(radio)
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



