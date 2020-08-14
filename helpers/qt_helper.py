# from PyQt5 import QtCore, QtWidgets   # use wrapper project QtPy instead
from qtpy import QtCore, QtWidgets
import numpy as np
import os
import contextlib
from helpers.util import INFO, WARN

__all__ = [
    "ClipboardMonitor",
]


# Origin: LuPY.Qt5._Qt5, Lu Jiakai
class ClipboardMonitor:
    FLAG_NO_RESULT = object()

    def __init__(self, modes="Path", qt_app=None, is_block_in_cb: bool = True):
        """
        :param modes: Type to Monitor on clipboard
            “Path" = File or Folder Path, return str:
            "Path_File" = File Path, return str
            “PathList” --> [str]: list of Paths to return
            ”Screen" --> np.ndarray: Screen Image in BGRA
            "Image" --> np.ndarray: File Image and Screen, File ext contains .bmp .jpg .png
        :param qt_app:
        :param is_block_in_cb: block in callbacks
        """
        modes = [modes] if not isinstance(modes, list) else modes
        assert [_ in ["Path", "Path_File", "PathList", "Screen", "Image"] for _ in modes], f"Invalid pMode: {modes}"

        # --- Params ---
        self.Mode = modes
        self.QtApp = qt_app
        self.IsBlockInCallback = is_block_in_cb

        # --- Values ---
        self.IsInCallback: bool = False
        self.QtClipboard = None
        self.Callback = None
        self.IsRunOnce = False
        self.CallResults = self.__class__.FLAG_NO_RESULT
        pass

    def run(self, cb: callable, onetime: bool = False):
        """
        :param cb: func like (mode:str, value:any). mode is defined in __init__()
        :param onetime: bool, False: run only once
        """
        self.Callback = cb
        self.IsRunOnce = onetime

        # --- Connect & Run
        INFO("")
        INFO("/////////////////////////////////////")
        INFO("// --- Begin Monitor Clipboard --- //")
        INFO("/////////////////////////////////////")

        # --- Run Monitor App
        self.QtApp = QtWidgets.QApplication([]) if self.QtApp is None else self.QtApp
        self.QtClipboard = self.QtApp.clipboard()
        self.QtClipboard.dataChanged.connect(self._slot_Clipboard_OnChanged)
        self.QtApp.exec()
        return self.CallResults if self.CallResults is not self.__class__.FLAG_NO_RESULT else None

    def _slot_Clipboard_OnChanged(self):
        try:
            # --- Only One Instance can be run
            if self.IsInCallback and self.IsBlockInCallback:
                return
            self.IsInCallback = True

            # --- Check Captured Data
            tMimeData: QtCore.QMimeData = self.QtClipboard.mimeData()

            # [A] Urls ---
            if len(tMimeData.urls()) > 0:
                # --- Get List
                tPathList = []
                for tUrl in tMimeData.urls():
                    tUrl: QtCore.QUrl = tUrl
                    tPath: str = tUrl.toLocalFile()
                    tPath = tPath.replace(r'\/'.replace(os.sep, ''), os.sep)

                    # Check exist
                    if (not os.path.isfile(tPath)) and (not os.path.isdir(tPath)):
                        continue
                    tPathList.append(tPath)

                # --- Distribute Message
                if len(tPathList) > 0:
                    if "Path_File" in self.Mode:
                        for tPath in tPathList:
                            if os.path.isfile(tPath):
                                self.CallResults = self.Callback("Path_File", tPath)

                    if "PathList" in self.Mode:
                        self.CallResults = self.Callback("PathList", tPathList)

                    if "Path" in self.Mode:
                        for tPath in tPathList:
                            self.CallResults = self.Callback("Path", tPath)

            # [B] Image Data
            if tMimeData.hasImage():
                tQImage = tMimeData.imageData()  # RGB32 with 0xffRRGGBB ===> BGRA in np
                tPtr = tQImage.constBits()
                tPtr.setsize(tQImage.byteCount())
                tMat = np.ndarray(buffer=tPtr, shape=[tQImage.height(), tQImage.width(), 4], dtype=np.uint8)

                if "Image" in self.Mode:
                    self.CallResults = self.Callback("Image", tMat)

        except Exception as e:
            WARN(f'Exception during clipboard event handling: {e}')
        finally:
            self.IsInCallback = False
            if self.IsRunOnce and self.CallResults is not self.__class__.FLAG_NO_RESULT:
                self.stop()  # only stop after a successful handling
        pass

    def stop(self):
        self.QtApp.quit()
        pass

    pass


@contextlib.contextmanager
def wait_signal(pSignal, pTimeout=None):
    """
    Block loop until signal emitted, or timeout (ms) elapses.
    """
    loop = QtCore.QEventLoop()
    pSignal.connect(loop.quit)

    yield

    if pTimeout is not None:
        QtCore.QTimer.singleShot(pTimeout, loop.quit)
    loop.exec_()
    pass
