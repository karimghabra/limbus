"""Launch the real MainWindow, let it scan for cameras, screenshot it, quit."""
import os
import sys

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

out = sys.argv[1]
app = QtWidgets.QApplication(sys.argv)
win = cr.MainWindow()
win.show()


def shot():
    win.grab().save(out)
    print("HAVE_PYLON:", cr.HAVE_PYLON, "| HAVE_ARAVIS:", cr.HAVE_ARAVIS)
    print("cameras found:", win.cameras)
    print("preview text:", win.preview.text().replace("\n", " / "))
    app.quit()


QtCore.QTimer.singleShot(3000, shot)
app.exec_()
