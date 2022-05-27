# Used to build the python MainWindow class to be used in main.py (user application).
# Builds mainwindow.py based on the files contained in the application_ui directory.

import sys, os
from PyQt5 import uic

# os.chdir(os.path.dirname(sys.argv[0]))
uic.compileUiDir('application_ui')