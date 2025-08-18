import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Qt
from PySide6.QtGui import QPixmap, QImage

# 第三方库（按需保留）
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import onnxruntime as ort


class Window:
    def __init__(self):
        # 加载UI文件
        self.ui = self.load_main_ui()
        # 初始化控件和事件绑定（示例）

    def load_main_ui(self):
        """加载 .ui 文件并返回窗口对象"""
        ui_path = Path(r'D:\untitled.ui')  # 改用 pathlib 处理路径
        if not ui_path.exists():
            raise FileNotFoundError(f"UI 文件不存在: {ui_path}")

        ui_file = QFile(str(ui_path))
        if not ui_file.open(QIODevice.ReadOnly):
            raise IOError(f"无法打开 UI 文件: {ui_path}")

        loader = QUiLoader()
        window = loader.load(ui_file)
        ui_file.close()

        if not window:
            raise RuntimeError("UI 文件加载失败，请检查文件格式是否正确")
        return window

    def show(self):
        """显示窗口"""
        self.ui.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局样式（可选）
    app.setStyle("Fusion")  # 更现代的样式

    window = Window()
    window.show()
    sys.exit(app.exec())