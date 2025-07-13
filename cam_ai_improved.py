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
    QMenu, QAction, QSystemTrayIcon, QStyle, QDesktopWidget, QInputDialog
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QThread, pyqtProperty

# AI 개선된 설정
CONFIG_FILE = "ai_cam_config.json"
RECORD_INTERVAL = 1800  # 30분
MAX_CONCURRENT_STREAMS = 12
STREAM_RESOLUTION = "640x480"
RECORD_RESOLUTION = "1920x1080"

# RTSP 최적화 설정
RTSP_TCP_OPTIONS = "rtsp_transport;tcp|stimeout;30000000|fflags;nobuffer|flags;low_delay|reorder_queue_size;0|max_delay;0|analyzeduration;2000000|probesize;1000000|max_probe_packets;1000|err_detect;ignore_err|skip_frame;nokey|skip_loop_filter;48|tune;zerolatency"
RTSP_UDP_OPTIONS = "rtsp_transport;udp|stimeout;30000000|fflags;nobuffer|flags;low_delay|reorder_queue_size;0|max_delay;0|analyzeduration;2000000|probesize;1000000|max_probe_packets;1000|err_detect;ignore_err|skip_frame;nokey|skip_loop_filter;48|tune;zerolatency"

class SmartCameraStream(threading.Thread):
    """AI 개선된 카메라 스트림 클래스"""
    
    def __init__(self, camera_id, label, camera_index, rtsp_url=None, record_dir=None, resolution="640x480"):
        super().__init__()
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.rtsp_url = rtsp_url
        self.label = label
        self.running = True
        self.cap = None
        self.record_dir = record_dir
        self.last_record_time = 0
        self.writer = None
        self.is_recording = False
        
        # AI 스마트 기능
        self.connection_quality = 0.0  # 0-100
        self.frame_drop_rate = 0.0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.error_count = 0
        self.reconnect_count = 0
        self.auto_reconnect = True
        
        # 해상도 설정
        self.stream_width, self.stream_height = map(int, resolution.split('x'))
        self.record_width, self.record_height = 1920, 1080
        
        # 성능 모니터링
        self.fps = 0.0
        self.actual_width = 0
        self.actual_height = 0
        
        self.daemon = True
        self.label.setText(f"📹 Camera {camera_index + 1}\n🔄 초기화 중...")

    def get_connection_status(self):
        """연결 상태를 점수로 반환 (0-100)"""
        if not self.cap or not self.cap.isOpened():
            return 0
        
        # 프레임 드롭률, 에러율, 재연결 횟수 등을 종합하여 점수 계산
        quality = 100.0
        
        if self.frame_drop_rate > 0.1:  # 10% 이상 드롭
            quality -= 30
        if self.error_count > 5:
            quality -= 20
        if self.reconnect_count > 3:
            quality -= 15
            
        return max(0, min(100, quality))

    def start_recording(self):
        """스마트 녹화 시작"""
        if not self.record_dir:
            return
            
        if self.writer:
            self.writer.release()
            
        os.makedirs(self.record_dir, exist_ok=True)
        filename = datetime.now().strftime("rec_%Y%m%d_%H%M%S.avi")
        filepath = os.path.join(self.record_dir, filename)
        
        # 최적화된 코덱 선택
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.writer = cv2.VideoWriter(filepath, fourcc, 30.0, (self.record_width, self.record_height))
            if not self.writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter(filepath, fourcc, 30.0, (self.record_width, self.record_height))
        except:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(filepath, fourcc, 30.0, (self.record_width, self.record_height))
            
        if self.writer.isOpened():
            self.last_record_time = time.time()
            print(f"🎬 Camera {self.camera_index + 1} 녹화 시작: {filename}")

    def stop_recording(self):
        """녹화 중지"""
        if self.writer:
            self.writer.release()
            self.writer = None
            print(f"⏹️ Camera {self.camera_index + 1} 녹화 중지")

    def run(self):
        """AI 개선된 스트림 처리"""
        reconnect_attempts = 0
        max_attempts = 3
        
        while self.running:
            if not self.cap or not self.cap.isOpened():
                if reconnect_attempts >= max_attempts:
                    self.label.setText(f"📹 Camera {self.camera_index + 1}\n❌ 연결 실패")
                    self.label.setStyleSheet("background-color: #ff4444; color: white; border-radius: 10px;")
                    break
                    
                self.label.setText(f"📹 Camera {self.camera_index + 1}\n🔄 재연결 중... ({reconnect_attempts + 1}/{max_attempts})")
                
                if self.cap:
                    self.cap.release()
                self.cap = None
                
                try:
                    if self.rtsp_url:
                        # DNS 확인
                        import socket
                        try:
                            if '@' in self.rtsp_url:
                                host = self.rtsp_url.split('@')[1].split(':')[0]
                            else:
                                host = self.rtsp_url.split('://')[1].split(':')[0]
                            socket.gethostbyname(host)
                        except:
                            pass
                        
                        # TCP 시도
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_TCP_OPTIONS
                        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                        
                        if not self.cap.isOpened():
                            # UDP 시도
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_UDP_OPTIONS
                            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                            
                    elif isinstance(self.camera_id, int):
                        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    
                    # 연결 대기
                    start_time = time.time()
                    while not self.cap.isOpened():
                        if time.time() - start_time > 30:
                            raise Exception("연결 시간 초과")
                        time.sleep(0.5)
                    
                    # 카메라 설정
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    
                    if self.fps <= 0.1 or self.fps > 60:
                        self.fps = 30.0
                    
                    # 최적 설정 적용
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.actual_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.actual_height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    reconnect_attempts = 0
                    self.reconnect_count += 1
                    
                except Exception as e:
                    print(f"Camera {self.camera_index + 1} 연결 실패: {e}")
                    reconnect_attempts += 1
                    time.sleep(1)
                    continue
            
            # 프레임 처리
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.error_count += 1
                    self.label.setText(f"📹 Camera {self.camera_index + 1}\n⚠️ 프레임 오류")
                    time.sleep(0.1)
                    continue
                
                # 성능 모니터링
                current_time = time.time()
                self.frame_count += 1
                
                if current_time - self.last_frame_time > 0:
                    self.frame_drop_rate = 1.0 - (self.frame_count / (self.fps * (current_time - self.last_frame_time)))
                
                # 녹화 처리
                if self.is_recording and self.record_dir:
                    now = time.time()
                    if self.writer is None or (now - self.last_record_time > RECORD_INTERVAL):
                        self.start_recording()
                    
                    if self.writer and frame is not None:
                        if frame.shape[1] != self.record_width or frame.shape[0] != self.record_height:
                            resized_frame = cv2.resize(frame, (self.record_width, self.record_height))
                        else:
                            resized_frame = frame
                        self.writer.write(resized_frame)
                
                # 화면 표시
                if frame is not None:
                    # 타임스탬프 추가
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # 상태 정보 추가
                    status_text = f"FPS: {self.fps:.1f} | Quality: {self.get_connection_status():.0f}%"
                    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 화면에 표시
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    pixmap = QPixmap.fromImage(qt_image).scaled(
                        self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.label.setPixmap(pixmap)
                    
                    # 연결 상태 업데이트
                    quality = self.get_connection_status()
                    if quality > 80:
                        self.label.setStyleSheet("background-color: #44ff44; color: white; border-radius: 10px;")
                    elif quality > 50:
                        self.label.setStyleSheet("background-color: #ffaa44; color: white; border-radius: 10px;")
                    else:
                        self.label.setStyleSheet("background-color: #ff4444; color: white; border-radius: 10px;")
                
                time.sleep(1.0 / self.fps if self.fps > 0 else 0.033)
                
            except Exception as e:
                self.error_count += 1
                print(f"Camera {self.camera_index + 1} 프레임 처리 오류: {e}")
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
    """시스템 모니터링 스레드"""
    system_update = pyqtSignal(dict)
    
    def run(self):
        while True:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.system_update.emit({
                'cpu': cpu_percent,
                'memory': memory.percent,
                'disk': disk.percent
            })
            
            time.sleep(2)

class SmartCameraViewer(QMainWindow):
    """AI 개선된 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 AI Smart Camera Monitor")
        self.setGeometry(100, 100, 1400, 900)
        
        # 데이터
        self.cameras = []
        self.streams = []
        self.is_recording = False
        self.expanded_camera = None
        
        # 시스템 모니터
        self.system_monitor = SystemMonitor()
        self.system_monitor.system_update.connect(self.update_system_status)
        self.system_monitor.start()
        
        # UI 초기화
        self.init_ui()
        self.load_config()
        
        # 시스템 트레이
        self.setup_system_tray()
        
        # 자동 저장 타이머
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_config)
        self.auto_save_timer.start(30000)  # 30초마다 자동 저장

    def init_ui(self):
        """UI 초기화"""
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 패널 (컨트롤)
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 오른쪽 패널 (카메라 뷰)
        right_panel = self.create_camera_panel()
        main_layout.addWidget(right_panel, 4)
        
        # 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 시스템 상태 표시
        self.cpu_label = QLabel("CPU: 0%")
        self.memory_label = QLabel("RAM: 0%")
        self.disk_label = QLabel("DISK: 0%")
        
        self.status_bar.addPermanentWidget(self.cpu_label)
        self.status_bar.addPermanentWidget(self.memory_label)
        self.status_bar.addPermanentWidget(self.disk_label)
        
        # 스타일 적용
        self.apply_dark_theme()

    def create_control_panel(self):
        """컨트롤 패널 생성"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(300)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 제목
        title = QLabel("🎛️ 제어 패널")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 빠른 시작 버튼
        quick_start_btn = QPushButton("🚀 빠른 시작")
        quick_start_btn.clicked.connect(self.quick_start)
        quick_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(quick_start_btn)
        
        # 녹화 컨트롤
        record_group = QGroupBox("🎬 녹화 제어")
        record_layout = QVBoxLayout()
        record_group.setLayout(record_layout)
        
        self.record_btn = QPushButton("⏺️ 녹화 시작")
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
        
        self.camera_list = QListWidget()
        self.camera_list.itemClicked.connect(self.on_camera_selected)
        camera_layout.addWidget(self.camera_list)
        
        # 카메라 추가/제거 버튼
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("➕ 추가")
        add_btn.clicked.connect(self.add_camera)
        remove_btn = QPushButton("➖ 제거")
        remove_btn.clicked.connect(self.remove_camera)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        camera_layout.addLayout(btn_layout)
        
        layout.addWidget(camera_group)
        
        # 설정
        settings_group = QGroupBox("⚙️ 설정")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)
        
        settings_btn = QPushButton("🔧 고급 설정")
        settings_btn.clicked.connect(self.open_settings)
        settings_layout.addWidget(settings_btn)
        
        layout.addWidget(settings_group)
        
        # 로그
        log_group = QGroupBox("📋 시스템 로그")
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
        """카메라 패널 생성"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 제목
        title = QLabel("📺 카메라 뷰")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 카메라 그리드
        self.camera_grid = QGridLayout()
        self.camera_grid.setSpacing(5)
        
        # 12개 카메라 라벨 생성
        for i in range(12):
            label = QLabel(f"📹 Camera {i + 1}\n⏳ 대기 중...")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("""
                QLabel {
                    background-color: #2b2b2b;
                    color: white;
                    border: 2px solid #444;
                    border-radius: 10px;
                    padding: 10px;
                    font-size: 12px;
                }
                QLabel:hover {
                    border-color: #666;
                }
            """)
            label.mouseDoubleClickEvent = self.make_double_click_handler(label, i)
            
            row, col = divmod(i, 4)
            self.camera_grid.addWidget(label, row, col)
            self.cameras.append(label)
        
        layout.addLayout(self.camera_grid)
        
        return panel

    def apply_dark_theme(self):
        """다크 테마 적용"""
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

    def setup_system_tray(self):
        """시스템 트레이 설정"""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # 트레이 메뉴
        tray_menu = QMenu()
        show_action = QAction("보이기", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        
        quit_action = QAction("종료", self)
        quit_action.triggered.connect(self.close)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def quick_start(self):
        """빠른 시작"""
        self.log_message("🚀 빠른 시작 실행 중...")
        
        # 기본 설정으로 자동 시작
        if not self.streams:
            self.start_all_cameras()
        
        self.log_message("✅ 빠른 시작 완료!")

    def start_all_cameras(self):
        """모든 카메라 시작"""
        self.log_message("📹 카메라 연결 시작...")
        
        for i, camera in enumerate(self.cameras):
            if i < len(self.cameras):
                # 기본 RTSP URL로 시작 (설정에서 로드)
                rtsp_url = f"rtsp://camera{i+1}.local:554/stream1"
                stream = SmartCameraStream(i, camera, i, rtsp_url=rtsp_url)
                self.streams.append(stream)
                stream.start()
                
                time.sleep(0.5)  # 순차 연결
        
        self.log_message("✅ 모든 카메라 연결 완료!")

    def toggle_recording(self):
        """녹화 토글"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """녹화 시작"""
        self.is_recording = True
        self.record_btn.setText("⏹️ 녹화 중지")
        self.record_status.setText("녹화 중...")
        self.record_btn.setStyleSheet("background-color: #f44336; color: white;")
        
        for stream in self.streams:
            stream.is_recording = True
        
        self.log_message("🎬 녹화 시작!")
        self.tray_icon.showMessage("녹화 시작", "모든 카메라에서 녹화가 시작되었습니다.")

    def stop_recording(self):
        """녹화 중지"""
        self.is_recording = False
        self.record_btn.setText("⏺️ 녹화 시작")
        self.record_status.setText("녹화 중지됨")
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        
        for stream in self.streams:
            stream.is_recording = False
        
        self.log_message("⏹️ 녹화 중지!")
        self.tray_icon.showMessage("녹화 중지", "모든 카메라에서 녹화가 중지되었습니다.")

    def add_camera(self):
        """카메라 추가"""
        # 간단한 카메라 추가 다이얼로그
        url, ok = QInputDialog.getText(self, "카메라 추가", "RTSP URL:")
        if ok and url:
            camera_id = len(self.streams)
            stream = SmartCameraStream(camera_id, self.cameras[camera_id], camera_id, rtsp_url=url)
            self.streams.append(stream)
            stream.start()
            
            self.log_message(f"➕ 카메라 {camera_id + 1} 추가됨: {url}")

    def remove_camera(self):
        """카메라 제거"""
        if self.streams:
            stream = self.streams.pop()
            stream.stop()
            self.log_message(f"➖ 카메라 {stream.camera_index + 1} 제거됨")

    def on_camera_selected(self, item):
        """카메라 선택됨"""
        camera_id = item.data(Qt.UserRole)
        self.log_message(f"📹 카메라 {camera_id + 1} 선택됨")

    def make_double_click_handler(self, label, index):
        """더블클릭 핸들러"""
        def handler(event):
            if self.expanded_camera == label:
                # 축소
                self.expanded_camera = None
                self.restore_grid()
            else:
                # 확대
                self.expanded_camera = label
                self.expand_camera(label)
        return handler

    def expand_camera(self, label):
        """카메라 확대"""
        # 그리드에서 모든 라벨 숨기기
        for i in range(self.camera_grid.count()):
            item = self.camera_grid.itemAt(i)
            if item.widget():
                item.widget().setVisible(False)
        
        # 선택된 라벨만 표시하고 크게
        label.setVisible(True)
        self.camera_grid.addWidget(label, 0, 0, 3, 4)

    def restore_grid(self):
        """그리드 복원"""
        # 모든 라벨 다시 표시
        for i, label in enumerate(self.cameras):
            label.setVisible(True)
            row, col = divmod(i, 4)
            self.camera_grid.addWidget(label, row, col, 1, 1)

    def update_system_status(self, status):
        """시스템 상태 업데이트"""
        self.cpu_label.setText(f"CPU: {status['cpu']:.1f}%")
        self.memory_label.setText(f"RAM: {status['memory']:.1f}%")
        self.disk_label.setText(f"DISK: {status['disk']:.1f}%")

    def log_message(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def open_settings(self):
        """설정 열기"""
        QMessageBox.information(self, "설정", "고급 설정 기능은 개발 중입니다.")

    def load_config(self):
        """설정 로드"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.log_message("📂 설정 파일 로드됨")
            except:
                self.log_message("⚠️ 설정 파일 로드 실패")
        else:
            self.log_message("📝 새 설정 파일 생성됨")

    def auto_save_config(self):
        """자동 설정 저장"""
        try:
            config = {
                'cameras': [],
                'settings': {
                    'recording_interval': RECORD_INTERVAL,
                    'max_streams': MAX_CONCURRENT_STREAMS
                }
            }
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except:
            pass

    def closeEvent(self, event):
        """프로그램 종료"""
        # 모든 스트림 중지
        for stream in self.streams:
            stream.stop()
        
        # 설정 저장
        self.auto_save_config()
        
        # 시스템 트레이에서 제거
        self.tray_icon.hide()
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("AI Smart Camera Monitor")
    
    # 아이콘 설정
    app.setWindowIcon(QIcon("camera_icon.png"))
    
    viewer = SmartCameraViewer()
    viewer.show()
    
    sys.exit(app.exec_()) 