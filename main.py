import sys
from PyQt5.QtWidgets import QApplication, QWizard, QWizardPage, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QSpinBox, QTextEdit, QRadioButton, QLabel, QLineEdit
from pages import FilePage, ModelPage


class CustomWizard(QWizard):
    def __init__(self):
        super().__init__()

        self.addPage(FilePage())
        self.addPage(ModelPage())
        #self.addPage(Page3())

        self.setWindowTitle("Build your Graph Neural Network")

def main():
    app = QApplication(sys.argv)
    wizard = CustomWizard()
    wizard.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()