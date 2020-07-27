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
    def __init__(self, pMode="Path", pQtApp=None, pIsBlockInCallback: bool = True):
        """
        :param pMode: Type to Monitor on clipboard
            “Path" = File or Folder Path, return str:
            "Path_File" = File Path, return str
            “PathList” --> [str]: list of Paths to return
            ”Screen" --> np.ndarray: Screen Image in BGRA
            "Image" --> np.ndarray: File Image and Screen, File ext contains .bmp .jpg .png
        :param pQtApp:
        :param pIsBlockInCallback:
        """
        pMode = [pMode] if not isinstance(pMode, list) else pMode
        assert [_ in ["Path", "Path_File", "PathList", "Screen", "Image"] for _ in pMode], f"Invalid pMode: {pMode}"

        # --- Params ---
        self.Mode = pMode
        self.QtApp = pQtApp
        self.IsBlockInCallback = pIsBlockInCallback

        # --- Values ---
        self.IsInCallback: bool = False
        self.QtClipboard = None
        self.Callback = None
        self.IsRunOnce = False
        self.CallResults = None
        pass

    def run(self, pCallback: callable, pIsRunOnce: bool = False):
        """
        :param pCallback: func like (pMode:str, pVal:any). pMode is defined in __init__()
        :param pIsRunOnce: bool, False: run only once
        """
        self.Callback = pCallback
        self.IsRunOnce = pIsRunOnce

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
        return self.CallResults

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
                    # tPath = tPath.replace("/", "\\")  # os compatibility
                    tPath = tPath.replace(r'\/'.replace(os.sep, ''), os.sep)

                    # Check exist
                    if (not os.path.isfile(tPath)) and (not os.path.isdir(tPath)):
                        continue
                    tPathList.append(tPath)

                # --- Distribute Message
                if "Path_File" in self.Mode:
                    for iPath in tPathList:
                        if os.path.isfile(iPath):
                            self.CallResults = self.Callback("Path_File", iPath)

                if "PathList" in self.Mode:
                    self.CallResults = self.Callback("PathList", tPathList)

                if "Path" in self.Mode:
                    for tPath in tPathList:
                        self.CallResults = self.Callback("Path", tPath)

                if "Image" in self.Mode:
                    # --- CV Import
                    from .. import OpenCV as LuPy_Cv2
                    for tPath in tPathList:
                        tExt: str = os.path.splitext(tPath)[-1]
                        if tExt.lower() in [".bmp", ".jpg", ".png"]:
                            tMat = LuPy_Cv2.safe_imread(tPath, -1)
                            self.CallResults = self.Callback("Image", tMat)

            # [B] Image Data
            if tMimeData.hasImage():
                tQImage = tMimeData.imageData()  # RGB32 with 0xffRRGGBB ===> BGRA in np
                tPtr = tQImage.constBits()
                tPtr.setsize(tQImage.byteCount())
                tMat = np.ndarray(buffer=tPtr, shape=[tQImage.height(), tQImage.width(), 4], dtype=np.uint8)

                # --- XOR
                if "Screen" in self.Mode:
                    self.CallResults = self.Callback("Image", tMat)

                elif "Image" in self.Mode:
                    self.CallResults = self.Callback("Image", tMat)

        except Exception as e:
            WARN(f'Exception during clipboard event handling: {e}')
        finally:
            self.IsInCallback = False
            if self.IsRunOnce:
                self.Stop()
        pass

    def Stop(self):
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
