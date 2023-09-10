import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFormLayout, QLabel, QCheckBox, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QDialogButtonBox, QDialog

class CustomizationWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph Neural Network Customization")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        self.node_features_group = QGroupBox("Node Features")
        self.node_features_layout = QFormLayout()
        self.node_features_group.setLayout(self.node_features_layout)

        self.graph_conv_group = QGroupBox("Graph Convolution Layer Type")
        self.graph_conv_attention_checkbox = QCheckBox("USE ATTENTION")
        self.graph_conv_layout = QFormLayout()
        self.graph_conv_group.setLayout(self.graph_conv_layout)

        self.pooling_group = QGroupBox("Pooling Strategies")
        self.pooling_layout = QFormLayout()
        self.pooling_group.setLayout(self.pooling_layout)

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
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CustomizationWindow()
    window.show()
    sys.exit(app.exec_())