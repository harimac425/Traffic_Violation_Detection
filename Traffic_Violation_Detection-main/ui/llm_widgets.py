"""
LLM UI Components: Model Selector and API Key Configuration - Material You Refinement
"""
import json
import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QMenu, QAction, QDialog, 
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QMessageBox, QFrame, QComboBox, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from pathlib import Path

# Add project root to sys.path if not present (for config import)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import config
except ImportError:
    # Fallback to local config if necessary
    config = None

SETTINGS_FILE = "settings.json"

class ApiKeyDialog(QDialog):
    """Refined Material Dialog for AI Brain Management"""
    def __init__(self, parent=None, model_to_set=None):
        super().__init__(parent)
        self.setWindowTitle("AI Brain Manager")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        self.model_to_set = model_to_set
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QLabel("<h2>🛡️ AI Brain Manager</h2>")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        desc = QLabel("Configure, test, and manage your AI inspection brains below.")
        desc.setObjectName("statusLabel")
        desc.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc)

        # Scroll Area for Provider Boxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        scroll_layout.setContentsMargins(0, 0, 10, 0)

        # --- Gemini Section ---
        self.gemini_box = self.create_provider_box(
            "Google Gemini", 
            "Enter Gemini API Key (AIza...)", 
            "https://aistudio.google.com/app/apikey",
            "gemini"
        )
        scroll_layout.addWidget(self.gemini_box)

        # --- OpenAI Section ---
        self.openai_box = self.create_provider_box(
            "OpenAI (GPT-4o)", 
            "Enter OpenAI Key (sk-...)", 
            "https://platform.openai.com/api-keys",
            "openai"
        )
        scroll_layout.addWidget(self.openai_box)

        # --- Custom Section ---
        self.custom_box = self.create_provider_box(
            "Custom Provider", 
            "Ollama / LM Studio Base URL", 
            None,
            "custom"
        )
        scroll_layout.addWidget(self.custom_box)
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # --- Rate Limits ---
        limit_box = QFrame()
        limit_box.setObjectName("controlBar")
        limit_l = QHBoxLayout(limit_box)
        limit_l.setContentsMargins(20, 10, 20, 10)
        
        self.rpm_spin = QComboBox()
        self.rpm_spin.addItems(["5 RPM", "15 RPM", "60 RPM", "1000 RPM"])
        
        limit_l.addWidget(QLabel("<b>Global Request Limit (RPM):</b>"))
        limit_l.addSpacing(10)
        limit_l.addWidget(self.rpm_spin)
        limit_l.addStretch()
        main_layout.addWidget(limit_box)
        
        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        save_btn = QPushButton("✅ Save & Update System")
        save_btn.setObjectName("startBtn")
        save_btn.setMinimumWidth(240)
        save_btn.clicked.connect(self.save_keys)
        
        cancel_btn = QPushButton("Close")
        cancel_btn.setMinimumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        main_layout.addLayout(btn_layout)
        
        self.load_keys()

    def create_provider_box(self, title, placeholder, link, key_type):
        """Create a stylized row for an LLM provider"""
        group = QFrame()
        group.setObjectName("providerBox")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        header_l = QHBoxLayout()
        title_lbl = QLabel(f"<b>{title}</b>")
        title_lbl.setStyleSheet("font-size: 15px; color: #D0BCFF;")
        header_l.addWidget(title_lbl)
        header_l.addStretch()
        
        if link:
            link_btn = QLabel(f"<a href='{link}' style='color: #DDE2F1; text-decoration: none;'>Get Key</a>")
            link_btn.setOpenExternalLinks(True)
            link_btn.setStyleSheet("font-size: 11px;")
            header_l.addWidget(link_btn)
        layout.addLayout(header_l)

        # Input Row
        input_l = QHBoxLayout()
        input_l.setSpacing(10)
        
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setEchoMode(QLineEdit.Password)
        line_edit.setReadOnly(True)
        line_edit.setMinimumHeight(44)
        input_l.addWidget(line_edit, stretch=4)
        
        if key_type == "custom":
            self.custom_url_input = line_edit
            self.custom_name_input = QLineEdit()
            self.custom_name_input.setPlaceholderText("Model Name")
            self.custom_name_input.setReadOnly(True)
            self.custom_name_input.setMinimumHeight(44)
            input_l.insertWidget(1, self.custom_name_input, stretch=2)
            
            self.custom_key_input = QLineEdit()
            self.custom_key_input.setPlaceholderText("Key (Opt)")
            self.custom_key_input.setEchoMode(QLineEdit.Password)
            self.custom_key_input.setReadOnly(True)
            self.custom_key_input.setMinimumHeight(44)
            input_l.insertWidget(2, self.custom_key_input, stretch=2)
            
        elif key_type == "gemini":
            self.gemini_key = line_edit
        elif key_type == "openai":
            self.openai_key = line_edit

        # Action Buttons - Standardized fixed width, reduced padding via object name
        test_btn = QPushButton("Test")
        test_btn.setMinimumWidth(75)
        test_btn.setStyleSheet("padding: 0 10px;") # Overwrite global 24px padding for tight fit
        test_btn.clicked.connect(lambda: self.test_provider(key_type))
        
        edit_btn = QPushButton("Edit")
        edit_btn.setMinimumWidth(75)
        edit_btn.setStyleSheet("padding: 0 10px;")
        edit_btn.clicked.connect(lambda: self.toggle_edit(key_type))
        
        del_btn = QPushButton("Del")
        del_btn.setMinimumWidth(75)
        del_btn.setObjectName("stopBtn")
        del_btn.setStyleSheet("padding: 0 10px;")
        del_btn.clicked.connect(lambda: self.delete_provider(key_type))
        
        input_l.addWidget(test_btn)
        input_l.addWidget(edit_btn)
        input_l.addWidget(del_btn)
        layout.addLayout(input_l)
        
        # Status Label
        status = QLabel("")
        status.setObjectName("statusLabel")
        if key_type == "gemini": self.gemini_status = status
        elif key_type == "openai": self.openai_status = status
        elif key_type == "custom": self.custom_status = status
        layout.addWidget(status)
        
        return group

    def toggle_edit(self, key_type):
        targets = []
        if key_type == "gemini": targets = [self.gemini_key]
        elif key_type == "openai": targets = [self.openai_key]
        elif key_type == "custom": targets = [self.custom_url_input, self.custom_name_input, self.custom_key_input]
        
        for t in targets:
            t.setReadOnly(not t.isReadOnly())
            if not t.isReadOnly():
                t.setFocus()

    def delete_provider(self, key_type):
        if QMessageBox.question(self, "Delete?", f"Clear settings for {key_type}?") == QMessageBox.Yes:
            if key_type == "gemini": self.gemini_key.clear()
            elif key_type == "openai": self.openai_key.clear()
            elif key_type == "custom":
                self.custom_url_input.clear()
                self.custom_name_input.clear()
                self.custom_key_input.clear()

    def test_provider(self, key_type):
        try:
            from src.llm import GeminiProvider, OpenAIProvider, CustomLLMProvider
        except ImportError:
            return
            
        status_lbl = getattr(self, f"{key_type}_status")
        status_lbl.setText("⏳ Testing connection...")
        status_lbl.repaint()
        
        try:
            if key_type == "gemini":
                key = self.gemini_key.text().strip()
                if not key: raise Exception("Key is empty")
                prov = GeminiProvider(key)
            elif key_type == "openai":
                key = self.openai_key.text().strip()
                if not key: raise Exception("Key is empty")
                prov = OpenAIProvider(key)
            else: # custom
                url = self.custom_url_input.text().strip()
                name = self.custom_name_input.text().strip()
                key = self.custom_key_input.text().strip()
                if not url: raise Exception("URL is empty")
                prov = CustomLLMProvider(key, url, name)
                
            success, msg = prov.test_connectivity()
            if success:
                status_lbl.setText(f"✅ {msg}")
                status_lbl.setStyleSheet("color: #D0BCFF; font-size: 11px;")
            else:
                status_lbl.setText(f"❌ {msg}")
                status_lbl.setStyleSheet("color: #FFB4AB; font-size: 11px;")
        except Exception as e:
            status_lbl.setText(f"❌ {str(e)}")
            status_lbl.setStyleSheet("color: #FFB4AB; font-size: 11px;")

    def load_keys(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    self.gemini_key.setText(data.get("GEMINI_API_KEY", ""))
                    self.openai_key.setText(data.get("OPENAI_API_KEY", ""))
                    self.custom_url_input.setText(data.get("CUSTOM_BASE_URL", ""))
                    self.custom_name_input.setText(data.get("CUSTOM_MODEL_NAME", ""))
                    self.custom_key_input.setText(data.get("CUSTOM_API_KEY", ""))
                    
                    rpm = data.get("LLM_MAX_RPM", 15)
                    if rpm <= 5: self.rpm_spin.setCurrentIndex(0)
                    elif rpm <= 15: self.rpm_spin.setCurrentIndex(1)
                    elif rpm <= 60: self.rpm_spin.setCurrentIndex(2)
                    else: self.rpm_spin.setCurrentIndex(3)
            except: pass
    
    def save_keys(self):
        data = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
            except: pass
            
        data["GEMINI_API_KEY"] = self.gemini_key.text().strip()
        data["OPENAI_API_KEY"] = self.openai_key.text().strip()
        data["CUSTOM_BASE_URL"] = self.custom_url_input.text().strip()
        data["CUSTOM_MODEL_NAME"] = self.custom_name_input.text().strip()
        data["CUSTOM_API_KEY"] = self.custom_key_input.text().strip()
        
        rpm_map = {0: 5, 1: 15, 2: 60, 3: 1000}
        data["LLM_MAX_RPM"] = rpm_map.get(self.rpm_spin.currentIndex(), 15)
        
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
            
        QMessageBox.information(self, "Success", "AI Settings updated successfully!")
        self.accept()

class ModelSelector(QWidget):
    """Refined Tooltip-style selector prioritizing hardware compute with conditional LLM visibility"""
    model_changed = pyqtSignal(str)
    device_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn = QPushButton("🧠 AI Brain: CPU")
        self.btn.setObjectName("brainSelector")
        self.btn.setMinimumHeight(40)
        self.btn.setStyleSheet("font-size: 11px; padding: 0 35px 0 15px; border-radius: 20px;")
        
        self.menu = QMenu(self)
        self.menu.setObjectName("violationMenu")
        self.btn.setMenu(self.menu)
        layout.addWidget(self.btn)
        
        self.refresh_menu()
        
        # Initial State
        if config:
            self.current_model = getattr(config, 'LLM_PROVIDER', 'local')
            self.current_device = getattr(config, 'DEVICE_TYPE', 'auto')
            self.update_button_text()

    def refresh_menu(self):
        """Build/Rebuild menu based on available keys"""
        self.menu.clear()
        
        # 1. Hardware Options (Always present)
        cpu_action = QAction("💻 Local CPU", self)
        cpu_action.triggered.connect(lambda: self.set_device_only("cpu"))
        self.menu.addAction(cpu_action)
        
        gpu_action = QAction("🚀 NVIDIA GPU", self)
        gpu_action.triggered.connect(lambda: self.set_device_only("cuda"))
        self.menu.addAction(gpu_action)
        
        auto_action = QAction("🤖 Auto-Detect Device", self)
        auto_action.triggered.connect(lambda: self.set_device_only("auto"))
        self.menu.addAction(auto_action)
        
        self.menu.addSeparator()
        
        # 2. Configured AI Brains (Conditional)
        has_ai = False
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    
                    if data.get("GEMINI_API_KEY"):
                        action = QAction("🧠 Google Gemini (Pro)", self)
                        action.triggered.connect(lambda: self.set_model("gemini"))
                        self.menu.addAction(action)
                        has_ai = True
                        
                    if data.get("OPENAI_API_KEY"):
                        action = QAction("🧠 OpenAI (GPT-4o)", self)
                        action.triggered.connect(lambda: self.set_model("openai"))
                        self.menu.addAction(action)
                        has_ai = True
                        
                    if data.get("CUSTOM_BASE_URL"):
                        action = QAction(f"🧠 {data.get('CUSTOM_MODEL_NAME', 'Custom LLM')}", self)
                        action.triggered.connect(lambda: self.set_model("custom"))
                        self.menu.addAction(action)
                        has_ai = True
            except: pass
            
        if has_ai:
            self.menu.addSeparator()
            
        # 3. Management
        setup_action = QAction("⚙️ Add / Configure AI Brains...", self)
        setup_action.triggered.connect(self.show_manager)
        self.menu.addAction(setup_action)

    def set_device_only(self, device_key):
        """Set compute but turn off LLM (requested Local behavior)"""
        self.current_model = "local"
        if config:
            config.LLM_PROVIDER = "local"
            config.DEVICE_TYPE = device_key
        
        self.save_settings({"LLM_PROVIDER": "local", "DEVICE_TYPE": device_key})
        self.device_changed.emit(device_key)
        self.model_changed.emit("local")
        self.update_button_text()

    def set_model(self, model_key):
        self.current_model = model_key
        if config:
            config.LLM_PROVIDER = model_key
        
        self.save_settings({"LLM_PROVIDER": model_key})
        self.update_button_text()
        self.model_changed.emit(model_key)

    def save_settings(self, updates):
        try:
            data = {}
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
            data.update(updates)
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except: pass

    def update_button_text(self):
        if self.current_model == "local" or self.current_model == "no_brain":
            # Show Device
            device_map = {"cpu": "CPU", "cuda": "GPU", "auto": "Auto"}
            dev = getattr(config, 'DEVICE_TYPE', 'auto')
            self.btn.setText(f"🧠 Brain: {device_map.get(dev, 'Local')}")
        else:
            # Show LLM
            names = {"gemini": "Gemini", "openai": "GPT-4o", "custom": "Custom"}
            self.btn.setText(f"🧠 Brain: {names.get(self.current_model, 'AI')}")

    def show_manager(self):
        diag = ApiKeyDialog(self.window())
        if diag.exec_() == QDialog.Accepted:
            self.refresh_menu()
        self.update_button_text()
