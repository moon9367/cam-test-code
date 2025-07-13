import sys
import cv2
import threading
import json
import os
import time
import psutil
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QGridLayout, QSizePolicy, QComboBox, QMessageBox,
    QDialog, QFormLayout, QFileDialog, QScrollArea, QCheckBox, QGroupBox,
    QMainWindow, QStatusBar, QProgressBar, QSlider, QFrame, QSplitter,
    QTabWidget, QTextEdit, QListWidget, QListWidgetItem, QToolButton,
    QMenu, QAction, QSystemTrayIcon, QStyle, QDesktopWidget, QInputDialog,
    QSpinBox, QDoubleSpinBox, QLabel, QButtonGroup, QRadioButton
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QThread

# 설정
CONFIG_FILE = "cam_config_v2.json"
RECORD_INTERVAL = 1800  # 30분
MAX_CONCURRENT_STREAMS = 12

class CameraConfigDialog(QDialog):
    """카메라 설정 다이얼로그"""
    
    def __init__(self, parent=None, camera_data=None):
        super().__init__(parent)
        self.setWindowTitle("카메라 설정")
        self.setModal(True)
        self.setFixedSize(500, 400)
        
        self.camera_data = camera_data or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 기본 정보
        basic_group = QGroupBox("기본 정보")
        basic_layout = QFormLayout()
        basic_group.setLayout(basic_layout)
        
        self.name_edit = QLineEdit(self.camera_data.get('name', ''))
        self.name_edit.setPlaceholderText("카메라 이름 (예: 입구 카메라)")
        basic_layout.addRow("카메라 이름:", self.name_edit)
        
        # 연결 타입
        self.connection_type = QComboBox()
        self.connection_type.addItems(["RTSP 스트림", "USB 카메라"])
        self.connection_type.currentTextChanged.connect(self.on_connection_type_changed)
        basic_layout.addRow("연결 타입:", self.connection_type)
        
        # RTSP URL
        self.rtsp_edit = QLineEdit(self.camera_data.get('rtsp_url', ''))
        self.rtsp_edit.setPlaceholderText("rtsp://username:password@ip:port/stream")
        basic_layout.addRow("RTSP URL:", self.rtsp_edit)
        
        # USB 카메라 인덱스
        self.usb_index = QSpinBox()
        self.usb_index.setRange(0, 10)
        self.usb_index.setValue(self.camera_data.get('usb_index', 0))
        basic_layout.addRow("USB 카메라 번호:", self.usb_index)
        
        layout.addWidget(basic_group)
        
        # 녹화 설정
        record_group = QGroupBox("녹화 설정")
        record_layout = QFormLayout()
        record_group.setLayout(record_layout)
        
        # 녹화 경로
        path_layout = QHBoxLayout()
        self.record_path_edit = QLineEdit(self.camera_data.get('record_path', './recordings'))
        self.record_path_edit.setPlaceholderText("녹화 파일 저장 경로")
        path_layout.addWidget(self.record_path_edit)
        
        self.browse_btn = QPushButton("찾아보기")
        self.browse_btn.clicked.connect(self.browse_record_path)
        path_layout.addWidget(self.browse_btn)
        
        record_layout.addRow("녹화 경로:", path_layout)
        
        # 해상도
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080", "3840x2160"])
        current_res = self.camera_data.get('resolution', '1280x720')
        index = self.resolution_combo.findText(current_res)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)
        record_layout.addRow("해상도:", self.resolution_combo)
        
        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.camera_data.get('fps', 30))
        record_layout.addRow("FPS:", self.fps_spin)
        
        # 활성화 여부
        self.enabled_check = QCheckBox("이 카메라 활성화")
        self.enabled_check.setChecked(self.camera_data.get('enabled', True))
        record_layout.addRow("", self.enabled_check)
        
        layout.addWidget(record_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("확인")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # 초기 상태 설정
        self.on_connection_type_changed(self.connection_type.currentText())
        
    def on_connection_type_changed(self, text):
        if text == "RTSP 스트림":
            self.rtsp_edit.setEnabled(True)
            self.usb_index.setEnabled(False)
        else:
            self.rtsp_edit.setEnabled(False)
            self.usb_index.setEnabled(True)
            
    def browse_record_path(self):
        path = QFileDialog.getExistingDirectory(self, "녹화 경로 선택")
        if path:
            self.record_path_edit.setText(path)
            
    def get_camera_data(self):
        return {
            'name': self.name_edit.text(),
            'connection_type': self.connection_type.currentText(),
            'rtsp_url': self.rtsp_edit.text(),
            'usb_index': self.usb_index.value(),
            'record_path': self.record_path_edit.text(),
            'resolution': self.resolution_combo.currentText(),
            'fps': self.fps_spin.value(),
            'enabled': self.enabled_check.isChecked()
        }

class CameraStream(threading.Thread):
    """개선된 카메라 스트림 클래스"""
    
    def __init__(self, camera_id, label, camera_data):
        super().__init__()
        self.camera_id = camera_id
        self.camera_data = camera_data
        self.label = label
        self.running = True
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.last_record_time = 0
        
        # 상태 정보
        self.connection_status = "disconnected"
        self.fps = 0.0
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = time.time()
        
        # 해상도 설정
        self.width, self.height = map(int, camera_data.get('resolution', '1280x720').split('x'))
        self.target_fps = camera_data.get('fps', 30)
        
        self.daemon = True
        self.update_status("초기화 중...")

    def update_status(self, status, color="#666666"):
        """상태 업데이트"""
        name = self.camera_data.get('name', f'Camera {self.camera_id + 1}')
        self.label.setText(f"{name}\n{status}")
        self.label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border: 2px solid #444;
                border-radius: 8px;
                padding: 10px;
                font-size: 11px;
                font-weight: bold;
            }}
        """)

    def start_recording(self):
        """녹화 시작"""
        if not self.camera_data.get('record_path'):
            return
            
        if self.writer:
            self.writer.release()
            
        record_path = self.camera_data['record_path']
        os.makedirs(record_path, exist_ok=True)
        
        filename = f"{self.camera_data.get('name', f'cam_{self.camera_id}')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        filepath = os.path.join(record_path, filename)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(filepath, fourcc, self.target_fps, (self.width, self.height))
            if not self.writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.writer = cv2.VideoWriter(filepath, fourcc, self.target_fps, (self.width, self.height))
        except:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(filepath, fourcc, self.target_fps, (self.width, self.height))
            
        if self.writer and self.writer.isOpened():
            self.last_record_time = time.time()
            print(f"🎬 {self.camera_data.get('name', f'Camera {self.camera_id + 1}')} 녹화 시작")

    def stop_recording(self):
        """녹화 중지"""
        if self.writer:
            self.writer.release()
            self.writer = None
            print(f"⏹️ {self.camera_data.get('name', f'Camera {self.camera_id + 1}')} 녹화 중지")

    def connect_camera(self):
        """카메라 연결"""
        try:
            if self.camera_data['connection_type'] == "RTSP 스트림":
                # RTSP 연결
                rtsp_url = self.camera_data['rtsp_url']
                if not rtsp_url:
                    return False
                    
                # RTSP 옵션 설정
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;30000000"
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                
                if not self.cap.isOpened():
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|stimeout;30000000"
                    self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    
            else:
                # USB 카메라 연결
                usb_index = self.camera_data['usb_index']
                self.cap = cv2.VideoCapture(usb_index, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            if not self.cap.isOpened():
                return False
                
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # 실제 값 확인
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            if actual_fps <= 0.1 or actual_fps > 60:
                actual_fps = self.target_fps
                
            self.fps = actual_fps
            return True
            
        except Exception as e:
            print(f"카메라 {self.camera_id + 1} 연결 실패: {e}")
            return False

    def run(self):
        """스트림 처리"""
        reconnect_attempts = 0
        max_attempts = 3
        
        while self.running:
            if not self.cap or not self.cap.isOpened():
                if reconnect_attempts >= max_attempts:
                    self.update_status("연결 실패", "#ff4444")
                    break
                    
                self.update_status(f"재연결 중... ({reconnect_attempts + 1}/{max_attempts})", "#ffaa44")
                
                if self.cap:
                    self.cap.release()
                self.cap = None
                
                if self.connect_camera():
                    reconnect_attempts = 0
                    self.update_status("연결됨", "#44ff44")
                else:
                    reconnect_attempts += 1
                    time.sleep(2)
                    continue
            
            # 프레임 처리
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.error_count += 1
                    if self.error_count > 10:
                        self.cap.release()
                        self.cap = None
                    time.sleep(0.1)
                    continue
                
                # 성능 모니터링
                current_time = time.time()
                self.frame_count += 1
                
                # 녹화 처리
                if self.is_recording:
                    now = time.time()
                    if self.writer is None or (now - self.last_record_time > RECORD_INTERVAL):
                        self.start_recording()
                    
                    if self.writer and frame is not None:
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            resized_frame = cv2.resize(frame, (self.width, self.height))
                        else:
                            resized_frame = frame
                        self.writer.write(resized_frame)
                
                # 화면 표시
                if frame is not None:
                    # 16:9 비율로 리사이즈
                    h, w = frame.shape[:2]
                    target_w = w
                    target_h = h
                    if w / h > 16/9:
                        target_w = int(h * 16 / 9)
                        target_h = h
                    else:
                        target_h = int(w * 9 / 16)
                        target_w = w
                    crop_x = (w - target_w) // 2
                    crop_y = (h - target_h) // 2
                    frame_16_9 = frame[crop_y:crop_y+target_h, crop_x:crop_x+target_w]
                    # 나머지 기존 코드와 동일하게 QImage 변환 및 setPixmap
                    rgb_image = cv2.cvtColor(frame_16_9, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.label.setPixmap(pixmap)
                
                time.sleep(1.0 / self.fps if self.fps > 0 else 0.033)
                
            except Exception as e:
                self.error_count += 1
                print(f"카메라 {self.camera_id + 1} 프레임 처리 오류: {e}")
                time.sleep(0.1)
        
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()

    def stop(self):
        """스트림 중지"""
        self.running = False
        self.is_recording = False
        self.stop_recording()
        self.join()

class SystemMonitor(QThread):
    """시스템 모니터링"""
    system_update = pyqtSignal(dict)
    
    def run(self):
        while True:
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')
                
                self.system_update.emit({
                    'cpu': cpu_percent,
                    'memory': memory.percent,
                    'disk': disk.percent
                })
            except:
                pass
            
            time.sleep(3)

class ImprovedCameraViewer(QMainWindow):
    """개선된 카메라 뷰어"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📹 개선된 카메라 모니터")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 데이터
        self.cameras = []
        self.streams = []
        self.is_recording = False
        
        # 시스템 모니터
        self.system_monitor = SystemMonitor()
        self.system_monitor.system_update.connect(self.update_system_status)
        self.system_monitor.start()
        
        # UI 초기화
        self.init_ui()
        self.load_config()
        
        # 자동 저장
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.save_config)
        self.auto_save_timer.start(30000)  # 30초마다 저장

    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 컨트롤 패널
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 오른쪽 카메라 패널
        right_panel = self.create_camera_panel()
        main_layout.addWidget(right_panel, 4)
        
        # 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 시스템 상태
        self.cpu_label = QLabel("CPU: 0%")
        self.memory_label = QLabel("RAM: 0%")
        self.disk_label = QLabel("DISK: 0%")
        
        self.status_bar.addPermanentWidget(self.cpu_label)
        self.status_bar.addPermanentWidget(self.memory_label)
        self.status_bar.addPermanentWidget(self.disk_label)
        
        # 스타일 적용
        self.apply_style()

    def create_control_panel(self):
        """컨트롤 패널 생성"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(350)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 제목
        title = QLabel("🎛️ 제어 패널")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 빠른 시작
        quick_group = QGroupBox("🚀 빠른 시작")
        quick_layout = QVBoxLayout()
        quick_group.setLayout(quick_layout)
        
        self.start_all_btn = QPushButton("모든 카메라 시작")
        self.start_all_btn.clicked.connect(self.start_all_cameras)
        quick_layout.addWidget(self.start_all_btn)
        
        self.stop_all_btn = QPushButton("모든 카메라 중지")
        self.stop_all_btn.clicked.connect(self.stop_all_cameras)
        quick_layout.addWidget(self.stop_all_btn)
        
        layout.addWidget(quick_group)
        
        # 녹화 제어
        record_group = QGroupBox("🎬 녹화 제어")
        record_layout = QVBoxLayout()
        record_group.setLayout(record_layout)
        
        self.record_btn = QPushButton("녹화 시작")
        self.record_btn.clicked.connect(self.toggle_recording)
        record_layout.addWidget(self.record_btn)
        
        self.record_status = QLabel("녹화 중지됨")
        self.record_status.setAlignment(Qt.AlignCenter)
        record_layout.addWidget(self.record_status)
        
        layout.addWidget(record_group)
        
        # 카메라 관리
        camera_group = QGroupBox("📹 카메라 관리")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        
        # 카메라 목록
        self.camera_list = QListWidget()
        self.camera_list.itemDoubleClicked.connect(self.edit_camera)
        camera_layout.addWidget(self.camera_list)
        
        # 카메라 버튼들
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("➕ 추가")
        add_btn.clicked.connect(self.add_camera)
        edit_btn = QPushButton("✏️ 편집")
        edit_btn.clicked.connect(self.edit_selected_camera)
        remove_btn = QPushButton("➖ 제거")
        remove_btn.clicked.connect(self.remove_camera)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(remove_btn)
        camera_layout.addLayout(btn_layout)
        
        layout.addWidget(camera_group)
        
        # 설정
        settings_group = QGroupBox("⚙️ 설정")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)
        
        save_btn = QPushButton("💾 설정 저장")
        save_btn.clicked.connect(self.save_config)
        settings_layout.addWidget(save_btn)
        
        load_btn = QPushButton("📂 설정 불러오기")
        load_btn.clicked.connect(self.load_config)
        settings_layout.addWidget(load_btn)
        
        layout.addWidget(settings_group)
        
        # 로그
        log_group = QGroupBox("📋 로그")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel

    def create_camera_panel(self):
        """카메라 패널 생성 (16:9, 4x3, 경계선 없음, 더블클릭 확대)"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        title = QLabel("📺 카메라 뷰")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.camera_grid = QGridLayout()
        self.camera_grid.setSpacing(0)  # 경계선 없음
        self.camera_labels = []
        self.expanded_label = None
        
        for i in range(12):
            label = QLabel(f"카메라 {i + 1}\n대기 중...")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(320, 180)  # 16:9 비율
            label.setMaximumSize(1920, 1080)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setStyleSheet("""
                QLabel {
                    background-color: #111;
                    color: white;
                    border: none;
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            label.setScaledContents(True)
            label.mouseDoubleClickEvent = self.make_double_click_handler(label, i)
            row, col = divmod(i, 4)
            self.camera_grid.addWidget(label, row, col)
            self.camera_labels.append(label)
        layout.addLayout(self.camera_grid)
        return panel

    def make_double_click_handler(self, label, index):
        def handler(event):
            if self.expanded_label == label:
                # 복귀
                self.expanded_label = None
                self.restore_grid()
            else:
                # 확대
                self.expanded_label = label
                self.expand_camera(label)
        return handler

    def expand_camera(self, label):
        # 모든 라벨 숨기기
        for l in self.camera_labels:
            l.setVisible(False)
        label.setVisible(True)
        # 전체 레이아웃에 1x1로 크게 추가
        self.camera_grid.addWidget(label, 0, 0, 3, 4)
        label.setMinimumSize(1280, 720)
        label.setMaximumSize(1920, 1080)
        label.setScaledContents(True)

    def restore_grid(self):
        # 원래 4x3 그리드로 복원
        for i, label in enumerate(self.camera_labels):
            label.setVisible(True)
            row, col = divmod(i, 4)
            self.camera_grid.addWidget(label, row, col, 1, 1)
            label.setMinimumSize(320, 180)
            label.setMaximumSize(1920, 1080)
            label.setScaledContents(True)

    def apply_style(self):
        """스타일 적용"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QListWidget {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #2b2b2b;
                color: white;
            }
        """)

    def start_all_cameras(self):
        """모든 카메라 시작"""
        self.log_message("📹 카메라 연결 시작...")
        
        for i, camera_data in enumerate(self.camera_configs):
            if camera_data.get('enabled', True):
                stream = CameraStream(i, self.cameras[i], camera_data)
                self.streams.append(stream)
                stream.start()
                time.sleep(0.5)  # 순차 연결
        
        self.log_message("✅ 모든 카메라 연결 완료!")

    def stop_all_cameras(self):
        """모든 카메라 중지"""
        self.log_message("🛑 모든 카메라 중지...")
        
        for stream in self.streams:
            stream.stop()
        self.streams.clear()
        
        # 라벨 초기화
        for i, label in enumerate(self.cameras):
            label.clear()
            label.setText(f"카메라 {i + 1}\n대기 중...")
            label.setStyleSheet("""
                QLabel {
                    background-color: #2b2b2b;
                    color: white;
                    border: 2px solid #444;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
        
        self.log_message("✅ 모든 카메라 중지 완료!")

    def toggle_recording(self):
        """녹화 토글"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """녹화 시작"""
        self.is_recording = True
        self.record_btn.setText("녹화 중지")
        self.record_status.setText("녹화 중...")
        self.record_btn.setStyleSheet("background-color: #f44336; color: white;")
        
        for stream in self.streams:
            stream.is_recording = True
        
        self.log_message("🎬 녹화 시작!")

    def stop_recording(self):
        """녹화 중지"""
        self.is_recording = False
        self.record_btn.setText("녹화 시작")
        self.record_status.setText("녹화 중지됨")
        self.record_btn.setStyleSheet("background-color: #444; color: white;")
        
        for stream in self.streams:
            stream.is_recording = False
        
        self.log_message("⏹️ 녹화 중지!")

    def add_camera(self):
        """카메라 추가"""
        dialog = CameraConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            camera_data = dialog.get_camera_data()
            self.camera_configs.append(camera_data)
            self.update_camera_list()
            self.log_message(f"➕ 카메라 추가: {camera_data.get('name', '이름 없음')}")

    def edit_selected_camera(self):
        """선택된 카메라 편집"""
        current_row = self.camera_list.currentRow()
        if current_row >= 0:
            self.edit_camera(self.camera_list.item(current_row))

    def edit_camera(self, item):
        """카메라 편집"""
        if not item:
            return
            
        index = item.data(Qt.UserRole)
        if 0 <= index < len(self.camera_configs):
            dialog = CameraConfigDialog(self, self.camera_configs[index])
            if dialog.exec_() == QDialog.Accepted:
                self.camera_configs[index] = dialog.get_camera_data()
                self.update_camera_list()
                self.log_message(f"✏️ 카메라 편집: {self.camera_configs[index].get('name', '이름 없음')}")

    def remove_camera(self):
        """카메라 제거"""
        current_row = self.camera_list.currentRow()
        if current_row >= 0 and current_row < len(self.camera_configs):
            camera_name = self.camera_configs[current_row].get('name', f'카메라 {current_row + 1}')
            reply = QMessageBox.question(self, "카메라 제거", f"'{camera_name}'을(를) 제거하시겠습니까?")
            
            if reply == QMessageBox.Yes:
                del self.camera_configs[current_row]
                self.update_camera_list()
                self.log_message(f"➖ 카메라 제거: {camera_name}")

    def update_camera_list(self):
        """카메라 목록 업데이트"""
        self.camera_list.clear()
        for i, camera_data in enumerate(self.camera_configs):
            name = camera_data.get('name', f'카메라 {i + 1}')
            enabled = "✅" if camera_data.get('enabled', True) else "❌"
            connection_type = camera_data.get('connection_type', 'Unknown')
            
            item = QListWidgetItem(f"{enabled} {name} ({connection_type})")
            item.setData(Qt.UserRole, i)
            self.camera_list.addItem(item)

    def update_system_status(self, status):
        """시스템 상태 업데이트"""
        self.cpu_label.setText(f"CPU: {status['cpu']:.1f}%")
        self.memory_label.setText(f"RAM: {status['memory']:.1f}%")
        self.disk_label.setText(f"DISK: {status['disk']:.1f}%")

    def log_message(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def load_config(self):
        """설정 로드"""
        self.camera_configs = []
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.camera_configs = data.get('cameras', [])
                    self.log_message("📂 설정 파일 로드됨")
            except Exception as e:
                self.log_message(f"⚠️ 설정 파일 로드 실패: {e}")
        else:
            self.log_message("📝 새 설정 파일 생성됨")
        
        self.update_camera_list()

    def save_config(self):
        """설정 저장"""
        try:
            config = {
                'cameras': self.camera_configs,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
                
            self.log_message("💾 설정 저장됨")
        except Exception as e:
            self.log_message(f"⚠️ 설정 저장 실패: {e}")

    def closeEvent(self, event):
        """프로그램 종료"""
        # 모든 스트림 중지
        for stream in self.streams:
            stream.stop()
        
        # 설정 저장
        self.save_config()
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("개선된 카메라 모니터")
    
    viewer = ImprovedCameraViewer()
    viewer.show()
    
    sys.exit(app.exec_()) 