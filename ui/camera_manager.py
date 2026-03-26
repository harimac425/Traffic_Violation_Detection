import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, 
    QPushButton, QInputDialog, QMessageBox, QLineEdit, QFormLayout, 
    QDialogButtonBox, QLabel, QWidget, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QColor

class CameraItemWidget(QFrame):
    """Custom widget for camera list items in the sidebar."""
    edit_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    duplicate_requested = pyqtSignal(int)
    connect_requested = pyqtSignal(int)

    def __init__(self, index, name, url, parent=None):
        super().__init__(parent)
        self.setObjectName("cameraItem")
        self.index = index
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # Info labels
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        name_label = QLabel(name)
        name_label.setObjectName("camName") # For styling specifically
        name_label.setWordWrap(True)
        
        url_label = QLabel(url)
        url_label.setObjectName("camUrl")
        url_label.setWordWrap(True)
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(url_label)
        
        layout.addLayout(info_layout, stretch=1)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.btn_dup = QPushButton("⧉") # Duplicate icon
        self.btn_dup.setToolTip("Duplicate")
        self.btn_dup.clicked.connect(lambda: self.duplicate_requested.emit(self.index))
        
        self.btn_edit = QPushButton("✎") # Edit icon
        self.btn_edit.setToolTip("Edit")
        self.btn_edit.clicked.connect(lambda: self.edit_requested.emit(self.index))
        
        self.btn_del = QPushButton("✕") # Delete icon
        self.btn_del.setToolTip("Delete")
        self.btn_del.clicked.connect(lambda: self.delete_requested.emit(self.index))
        
        btn_layout.addWidget(self.btn_dup)
        btn_layout.addWidget(self.btn_edit)
        btn_layout.addWidget(self.btn_del)
        
        layout.addLayout(btn_layout)
        
        # Make the whole widget clickable for connection
        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.connect_requested.emit(self.index)
        super().mousePressEvent(event)

class CameraManagerDialog(QDialog):
    """Dialog for managing and selecting RTSP/HTTP IP cameras."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cameraManager")
        self.setWindowTitle("Camera Manager")
        self.setMinimumSize(450, 500)
        self.selected_camera_url = None
        
        # Path to cameras config
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        self.config_file = data_dir / "cameras.json"
        
        self.cameras = self._load_cameras()
        self.init_ui()
        
    def _load_cameras(self):
        """Load cameras from JSON file. Returns list of dicts."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load cameras.json: {e}")
        
        # Default fallback
        return [
            {"name": "Default Webcam", "url": "0"}
        ]
        
    def _save_cameras(self):
        """Save cameras to JSON file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.cameras, f, indent=4)
        except Exception as e:
            print(f"Failed to save cameras.json: {e}")
            
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("Manage Cameras")
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #D0BCFF; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # List of cameras
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("cameraList")
        self.refresh_list()
        layout.addWidget(self.list_widget)
        
        # Action Buttons
        btn_row = QHBoxLayout()
        
        btn_add = QPushButton("+ Add Camera")
        btn_add.setObjectName("openBtn")
        btn_add.clicked.connect(self.on_add)
        btn_row.addWidget(btn_add)
        
        layout.addLayout(btn_row)
        
        # Bottom controls
        btn_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        btn_connect = QPushButton("Select & Connect")
        btn_connect.setObjectName("startBtn")
        btn_connect.setDefault(True)
        btn_connect.clicked.connect(self.on_connect)
        btn_box.addButton(btn_connect, QDialogButtonBox.AcceptRole)
        
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def refresh_list(self):
        """Refresh the UI list based on self.cameras"""
        self.list_widget.clear()
        for idx, cam in enumerate(self.cameras):
            item = QListWidgetItem(self.list_widget)
            widget = CameraItemWidget(idx, cam['name'], cam['url'])
            widget.edit_requested.connect(self.on_edit)
            widget.delete_requested.connect(self.on_remove)
            widget.duplicate_requested.connect(self.on_duplicate)
            widget.connect_requested.connect(self.select_and_accept)
            
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
            
    def select_and_accept(self, idx):
        self.list_widget.setCurrentRow(idx)
        self.on_connect()

    def get_selected_index(self):
        return self.list_widget.currentRow()

    def on_add(self):
        dialog = CameraEditDialog(self)
        if dialog.exec_():
            new_cam = dialog.get_data()
            if new_cam['name'] and new_cam['url']:
                self.cameras.append(new_cam)
                self._save_cameras()
                self.refresh_list()

    def on_edit(self, idx):
        cam = self.cameras[idx]
        dialog = CameraEditDialog(self, cam['name'], cam['url'])
        if dialog.exec_():
            updated_cam = dialog.get_data()
            if updated_cam['name'] and updated_cam['url']:
                self.cameras[idx] = updated_cam
                self._save_cameras()
                self.refresh_list()

    def on_remove(self, idx):
        reply = QMessageBox.question(
            self, 'Remove Camera', 
            f"Remove '{self.cameras[idx]['name']}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.cameras.pop(idx)
            self._save_cameras()
            self.refresh_list()
            
    def on_duplicate(self, idx):
        cam = self.cameras[idx].copy()
        cam['name'] += " (Copy)"
        self.cameras.insert(idx + 1, cam)
        self._save_cameras()
        self.refresh_list()

    def on_connect(self):
        idx = self.get_selected_index()
        if idx >= 0:
            self.selected_camera_url = self.cameras[idx]['url']
            self.accept()
        else:
            QMessageBox.warning(self, "Selection Required", "Please select a camera.")


class CameraEditDialog(QDialog):
    """Sub-dialog for entering camera details."""
    def __init__(self, parent=None, name="", url=""):
        super().__init__(parent)
        self.setWindowTitle("Camera Details" if not name else "Edit Camera")
        self.setMinimumWidth(350)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        form = QFormLayout()
        form.setSpacing(15)
        
        self.name_input = QLineEdit(name)
        self.name_input.setPlaceholderText("e.g. Traffic Cam 1")
        
        self.url_input = QLineEdit(url)
        self.url_input.setPlaceholderText("rtsp://... or http://... or 0")
        
        form.addRow("<b>Name:</b>", self.name_input)
        form.addRow("<b>URL/RTSP:</b>", self.url_input)
        
        layout.addLayout(form)
        layout.addSpacing(10)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
    def get_data(self):
        return {
            "name": self.name_input.text().strip(),
            "url": self.url_input.text().strip()
        }
