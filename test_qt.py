import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setGeometry(100, 100, 400, 300)
    window.setWindowTitle('PyQt5 Test')
    
    label = QLabel('Hello PyQt5!', window)
    label.move(150, 120)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 