import sys
from PyQt5.QtWidgets import QApplication, QWizard, QWizardPage, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QSpinBox, QTextEdit, QRadioButton, QLabel, QLineEdit
from PyQt5.QtGui import QPixmap
from pages import FilePage, ModelPage, HyperParamDialog


class CustomWizard(QWizard):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Build your Graph Neural Network")

        #logo = QPixmap("logo.png").scaled(100,70)
        banner = QPixmap("banner.png").scaled(125,550)
        #self.setPixmap(QWizard.LogoPixmap, logo)
        self.setPixmap(QWizard.WatermarkPixmap, banner)
        self.setWizardStyle(QWizard.ClassicStyle)

        self.addPage(FilePage())
        self.addPage(ModelPage())
        self.addPage(HyperParamDialog())



def main():
    app = QApplication(sys.argv)
    wizard = CustomWizard()
    wizard.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()