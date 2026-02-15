from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QProgressBar
from PyQt6.QtCore import pyqtSignal

class ControlPanel(QWidget):
    new_mission_clicked = pyqtSignal()
    load_database_clicked = pyqtSignal()
    start_tracking_clicked = pyqtSignal()
    stop_tracking_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.btn_new_mission = QPushButton("Створення нової місії")
        self.btn_load_db = QPushButton("Вибір бази даних HDF5")
        self.btn_calibrate = QPushButton("Калібрування GPS")
        
        self.btn_start = QPushButton("Почати відстеження")
        self.btn_stop = QPushButton("Зупинити відстеження")
        self.btn_stop.setEnabled(False)
        
        self.status_label = QLabel("Статус: Очікування")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        layout.addWidget(self.btn_new_mission)
        layout.addWidget(self.btn_load_db)
        layout.addWidget(self.btn_calibrate)
        layout.addSpacing(20)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        self.connect_signals()
        
    def connect_signals(self):
        self.btn_new_mission.clicked.connect(self.new_mission_clicked)
        self.btn_load_db.clicked.connect(self.load_database_clicked)
        self.btn_calibrate.clicked.connect(self.calibrate_clicked)
        self.btn_start.clicked.connect(self.start_tracking_clicked)
        self.btn_stop.clicked.connect(self.stop_tracking_clicked)
        
    def update_status(self, message: str):
        self.status_label.setText(f"Статус: {message}")
    
    def update_progress(self, value: int):
        self.progress_bar.setValue(value)