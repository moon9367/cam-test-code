import sys
import cv2
import threading
import json
import os
import time
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QGridLayout, QSizePolicy, QComboBox, QMessageBox,
    QDialog, QFormLayout, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal

# Configuration file for saving camera settings
CONFIG_FILE = "rtsp_config.json"
# Maximum size for QWidget, used for expanding labels
QWIDGETSIZE_MAX = 16777215
# Interval for automatically segmenting video recordings (30 minutes in seconds)
RECORD_INTERVAL = 1800

# Supported resolutions for selection (ordered by width, then height)
AVAILABLE_RESOLUTIONS = [
    "640x480", "1280x720", "1920x1080", "2048x1080", "2560x1440", "2880x1620", "4096x2160"
]

# TARGET_DISPLAY_WIDTH = 1920  # 고정할 너비
# TARGET_DISPLAY_HEIGHT = 1080 # 고정할 높이

class CameraStream(threading.Thread):
    """
    A thread class to handle capturing video from a camera (USB or RTSP)
    and displaying it on a QLabel, with optional recording capabilities.
    """
    def __init__(self, camera_identifier, label, camera_index, rtsp_url=None, record_base_dir=None, resolution_str="1920x1080"):
        super().__init__()
        self.camera_identifier = camera_identifier
        self.camera_index = camera_index
        self.rtsp_url = rtsp_url
        self.label = label
        self.running = True
        self.cap = None
        self.record_base_dir = record_base_dir
        self.camera_record_dir = None
        self.last_record_time = 0
        self.writer = None
        self.is_recording_active = False

        self.resolution_str = resolution_str
        self.desired_width = 1920
        self.desired_height = 1080

        # 카메라가 실제로 보고한 최대 해상도 및 FPS를 저장할 변수 (초기값)
        self.actual_max_width = 0
        self.actual_max_height = 0
        self.actual_max_fps = 0.0

        # 현재 카메라가 실제로 캡처하고 있는 해상도 (프레임 획득 시 갱신)
        self.current_capture_width = 0
        self.current_capture_height = 0
        self.current_capture_fps = 0.0

        # 화면 업데이트 빈도 조절을 위한 변수 추가
        self.last_display_time = time.time()

        try:
            width_str, height_str = self.resolution_str.split('x')
            self.desired_width = int(width_str)
            self.desired_height = int(height_str)
        except ValueError:
            print(f"경고: CAM {self.camera_index + 1}에 대한 잘못된 해상도 문자열: {self.resolution_str}. 기본값 1920x1080으로 설정합니다.")
            self.desired_width = 1920
            self.desired_height = 1080

        self.label.setText(f"Camera {self.camera_index + 1}\n신호 없음")

    def start_recording_segment(self):
        if self.record_base_dir:
            self.camera_record_dir = os.path.join(self.record_base_dir, f"cam{self.camera_index + 1}")
            os.makedirs(self.camera_record_dir, exist_ok=True)
        else:
            print(f"경고: CAM {self.camera_index + 1}에 대한 녹화 디렉토리가 설정되지 않았습니다. 세그먼트를 시작할 수 없습니다.")
            return

        if self.writer:
            self.writer.release()

        # 녹화는 사용자가 설정한 해상도로 진행
        record_width = self.desired_width
        record_height = self.desired_height
        
        # 녹화 FPS를 30.0으로 강제 고정
        record_fps = 30.0 
        print(f"CAM {self.camera_index + 1}: 녹화 FPS를 {record_fps:.2f}로 강제 고정하여 녹화 시작.")
        
        filename = datetime.now().strftime("rec_%Y%m%d_%H%M%S.mp4") # 확장자 변경
        filepath = os.path.join(self.camera_record_dir, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') # MPEG-4 Video (더 범용적)

        self.writer = cv2.VideoWriter(filepath, fourcc, record_fps, (record_width, record_height))
        self.last_record_time = time.time()
        
        if not self.writer.isOpened():
            print(f"오류: CAM {self.camera_index + 1}: VideoWriter를 열 수 없습니다. 코덱 또는 경로 문제일 수 있습니다. (경로: {filepath}, 해상도: {record_width}x{record_height}, FPS: {record_fps})")
            self.writer = None 
        else:
            print(f"CAM {self.camera_index + 1}: 녹화 세그먼트 시작: {filepath} ({record_width}x{record_height} @ {record_fps:.2f}fps)")

    def stop_recording_segment(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            print(f"CAM {self.camera_index + 1}: 녹화 세그먼트 중지.")

    def set_recording_active(self, active):
        self.is_recording_active = active
        if not active:
            self.stop_recording_segment()

    def run(self):
        try:
            reconnection_attempts = 0
            max_reconnection_attempts = 10 # 최대 재시도 횟수
            reconnection_delay = 2 # 재시도 간 지연 시간 (초)

            while self.running:
                if not self.cap or not self.cap.isOpened():
                    self.label.setText(f"CAM {self.camera_index + 1}: 연결 끊김, 재연결 시도 중... ({reconnection_attempts}/{max_reconnection_attempts})")
                    
                    if reconnection_attempts >= max_reconnection_attempts:
                        self.label.setText(f"CAM {self.camera_index + 1}: 연결 실패 (재시도 한도 초과)")
                        self.running = False
                        break
                    
                    if self.cap:
                        self.cap.release()
                    self.cap = None 

                    try: 
                        if self.rtsp_url:
                            self.cap = cv2.VideoCapture(f"{self.rtsp_url}?buffer_size=3", cv2.CAP_FFMPEG)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) 
                        elif isinstance(self.camera_identifier, int):
                            self.cap = cv2.VideoCapture(self.camera_identifier, cv2.CAP_DSHOW)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                        
                        start_time = time.time()
                        while not self.cap.isOpened():
                            if time.time() - start_time > 15: 
                                raise Exception("카메라 연결 시간 초과")
                            time.sleep(0.5)

                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999) 
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
                        self.cap.set(cv2.CAP_PROP_FPS, 9999) 
                        self.actual_max_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.actual_max_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.actual_max_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        if self.actual_max_fps <= 0.1: self.actual_max_fps = 30.0 

                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
                        self.cap.set(cv2.CAP_PROP_FPS, 30.0) 
                        
                        self.current_capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.current_capture_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.current_capture_fps = self.cap.get(cv2.CAP_PROP_FPS) 
                        if self.current_capture_fps <= 0.1: self.current_capture_fps = 30.0

                        print(f"CAM {self.camera_index + 1}: 재연결 성공. 실제 캡처: {self.current_capture_width}x{self.current_capture_height} @ {self.current_capture_fps:.2f}fps (요청: {self.desired_width}x{self.desired_height})")
                        reconnection_attempts = 0

                    except Exception as e:
                        print(f"CAM {self.camera_index + 1}: 재연결 시도 {reconnection_attempts + 1}회 실패: {e}")
                        reconnection_attempts += 1
                        time.sleep(reconnection_delay)
                    continue
                
                ret, frame = self.cap.read() 
                
                if not ret or frame is None:
                    self.label.setText(f"CAM {self.camera_index + 1}: 프레임 획득 실패")
                    if ret:
                        time.sleep(0.01)
                    else:
                        print(f"CAM {self.camera_index + 1}: 프레임 획득 실패, 재연결 시도 필요.")
                        self.cap.release()
                        self.cap = None 
                    continue
                
                h_frame, w_frame, _ = frame.shape 
                
                if self.current_capture_width == 0 or self.current_capture_height == 0:
                    self.current_capture_width = w_frame
                    self.current_capture_height = h_frame

                base_height_for_font_scale = 1080.0 
                font_scale = (h_frame / base_height_for_font_scale) * 1.0 
                
                min_font_scale_threshold = (480 / base_height_for_font_scale) * 1.0 * 0.8 
                font_scale = max(font_scale, min_font_scale_threshold)
                
                font_thickness = max(1, int(font_scale * 2)) 

                timestamp = datetime.now().strftime("%Y-%m-%d %a. %H:%M:%S")
                text_x = int(w_frame * 0.01) 
                text_y = int(h_frame * 0.03) 

                cv2.putText(frame, timestamp, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (255, 255, 255), font_thickness)

                if self.is_recording_active and self.record_base_dir:
                    now = time.time()
                    if self.writer is None or (now - self.last_record_time > RECORD_INTERVAL):
                        self.start_recording_segment()
                    
                    if self.writer and frame is not None:
                        original_rec_aspect = w_frame / h_frame
                        target_rec_width = self.desired_width
                        target_rec_height = int(target_rec_width / original_rec_aspect)

                        if target_rec_height > self.desired_height:
                            target_rec_height = self.desired_height
                            target_rec_width = int(target_rec_height * original_rec_aspect)

                        if target_rec_width <= 0 or target_rec_height <= 0:
                            target_rec_width = 1920
                            target_rec_height = 1080
                            
                        resized_frame_for_recording = cv2.resize(frame, (target_rec_width, target_rec_height), interpolation=cv2.INTER_AREA)
                        
                        if resized_frame_for_recording.shape[1] != self.desired_width or resized_frame_for_recording.shape[0] != self.desired_height:
                            resized_frame_for_recording = cv2.resize(resized_frame_for_recording, (self.desired_width, self.desired_height), interpolation=cv2.INTER_LINEAR)

                        self.writer.write(resized_frame_for_recording)
                elif not self.is_recording_active and self.writer is not None:
                    self.stop_recording_segment()

                current_time = time.time()
                display_interval = 1.0 / 30.0

                if (current_time - self.last_display_time) > display_interval:
                    # 라벨의 크기가 아니라, 1920x1080 비율을 목표로 합니다.
                    # 이전에 TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT 상수로 선언했던 부분을
                    # 클래스 멤버 변수로 옮기고, 16:9 비율을 유지하며 QLabel 크기에 맞추는 로직을 적용합니다.
                    
                    # 목표 해상도 및 비율 (1920x1080)
                    target_fixed_width = 1920
                    target_fixed_height = 1080
                    target_aspect_ratio = target_fixed_width / target_fixed_height

                    # 현재 QLabel의 크기
                    label_w = self.label.width()
                    label_h = self.label.height()

                    # QLabel의 크기 내에서 16:9 비율을 유지하며 최대화
                    display_w_in_label = label_w
                    display_h_in_label = int(display_w_in_label / target_aspect_ratio)

                    if display_h_in_label > label_h:
                        display_h_in_label = label_h
                        display_w_in_label = int(display_h_in_label * target_aspect_ratio)

                    # 최소 해상도 보장 (너무 작아지지 않도록)
                    min_display_width = 640
                    min_display_height = 480
                    
                    if display_w_in_label < min_display_width or display_h_in_label < min_display_height:
                        # 최소 해상도를 기준으로 다시 비율 계산
                        if min_display_width / min_display_height > target_aspect_ratio: # 최소 너비가 너무 작을 때
                            display_h_in_label = min_display_height
                            display_w_in_label = int(display_h_in_label * target_aspect_ratio)
                        else: # 최소 높이가 너무 작을 때
                            display_w_in_label = min_display_width
                            display_h_in_label = int(display_w_in_label / target_aspect_ratio)
                    
                    if display_w_in_label <= 0 or display_h_in_label <= 0:
                        display_w_in_label = 640
                        display_h_in_label = 480
                        print(f"경고: CAM {self.camera_index + 1} 표시 해상도가 0이 되어 기본값 {display_w_in_label}x{display_h_in_label}로 설정.")

                    # 최종적으로 표시할 프레임 리사이징
                    # 여기서 원본 프레임(frame)을 원하는 16:9 비율의 display_w_in_label, display_h_in_label로 리사이징합니다.
                    # 이 때 원본 프레임이 16:9가 아니라면, 찌그러짐 없이 16:9에 맞추기 위해 크롭 또는 레터박스가 발생할 수 있습니다.
                    # cv2.resize는 원본 종횡비를 무시하고 강제로 대상 크기로 늘리므로,
                    # 여기서는 원본 프레임을 먼저 16:9 비율의 임시 프레임으로 변환하는 로직이 필요합니다.
                    
                    # 원본 프레임의 가로세로 비율
                    original_frame_aspect = w_frame / h_frame

                    if original_frame_aspect > target_aspect_ratio:
                        # 원본이 16:9보다 가로가 길다 -> 세로를 16:9에 맞추고 가로는 크롭
                        new_h = h_frame
                        new_w = int(new_h * target_aspect_ratio)
                        x_offset = (w_frame - new_w) // 2
                        cropped_frame = frame[:, x_offset:x_offset + new_w]
                    elif original_frame_aspect < target_aspect_ratio:
                        # 원본이 16:9보다 세로가 길다 -> 가로를 16:9에 맞추고 세로는 크롭
                        new_w = w_frame
                        new_h = int(new_w / target_aspect_ratio)
                        y_offset = (h_frame - new_h) // 2
                        cropped_frame = frame[y_offset:y_offset + new_h, :]
                    else:
                        # 원본이 이미 16:9
                        cropped_frame = frame

                    # 이제 16:9 비율로 맞춰진 cropped_frame을 QLabel 크기에 맞춰 최종 리사이징
                    frame_display_for_screen = cv2.resize(cropped_frame, (display_w_in_label, display_h_in_label), interpolation=cv2.INTER_LINEAR)


                    rgb_image = cv2.cvtColor(frame_display_for_screen, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    pixmap = QPixmap.fromImage(qt_image).scaled(
                        self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation 
                    )
                    self.label.setPixmap(pixmap)
                    self.last_display_time = current_time
                
                time.sleep(0.001)

        except Exception as e:
            self.label.setText(f"CAM {self.camera_index + 1} 오류: {str(e)}")
            print(f"CAM {self.camera_index + 1} 스레드 오류: {e}")
        finally:
            if self.cap:
                self.cap.release()
            if self.writer:
                self.writer.release()
            print(f"CAM {self.camera_index + 1} 스레드 종료.")

    def stop(self):
        self.running = False
        self.set_recording_active(False)
        self.join()

class SettingsDialog(QDialog):
    def __init__(self, parent, cam_sources, camera_capabilities):
        super().__init__(parent)
        self.setWindowTitle("카메라 설정")
        self.cam_sources = cam_sources
        self.camera_capabilities = camera_capabilities
        layout = QFormLayout()
        self.inputs = [] 

        for i in range(6):
            type_combo = QComboBox()
            type_combo.addItems(["USB", "RTSP"])
            
            value_edit = QLineEdit() 

            resolution_combo = QComboBox()
            max_w, max_h, max_fps = self.camera_capabilities[i].get('max_width', 0), \
                                    self.camera_capabilities[i].get('max_height', 0), \
                                    self.camera_capabilities[i].get('max_fps', 0.0)
            
            if max_fps <= 0.1:
                max_fps = 30.0

            filtered_resolutions = []
            for res_str in AVAILABLE_RESOLUTIONS:
                w, h = map(int, res_str.split('x'))
                if w <= max_w and h <= max_h:
                    filtered_resolutions.append(res_str)
            
            actual_res_str = f"{max_w}x{max_h}"
            if max_w > 0 and max_h > 0 and actual_res_str not in filtered_resolutions:
                inserted = False
                for j, existing_res in enumerate(filtered_resolutions):
                    ew, eh = map(int, existing_res.split('x'))
                    if (max_w < ew) or (max_w == ew and max_h < eh):
                        filtered_resolutions.insert(j, actual_res_str)
                        inserted = True
                        break
                if not inserted:
                    filtered_resolutions.append(actual_res_str)
            
            final_resolutions_sorted = sorted(list(set(filtered_resolutions)), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))

            if not final_resolutions_sorted:
                final_resolutions_sorted.append("640x480")

            resolution_combo.addItems(final_resolutions_sorted)
            
            fps_label = QLabel(f"최대 FPS: {max_fps:.2f}")
            
            layout.addRow(f"CAM{i+1} 유형", type_combo)
            layout.addRow(f"CAM{i+1} 입력값", value_edit)
            
            res_fps_layout = QHBoxLayout()
            res_fps_layout.addWidget(resolution_combo)
            res_fps_layout.addWidget(fps_label)
            res_fps_layout.addStretch(1) 

            layout.addRow(f"CAM{i+1} 해상도", res_fps_layout)
            self.inputs.append((type_combo, value_edit, resolution_combo, fps_label))

        self.record_path_edit = QLineEdit()
        record_path_btn = QPushButton("녹화 폴더 선택")
        record_path_btn.clicked.connect(self.select_record_dir)
        layout.addRow("녹화 저장 경로", self.record_path_edit)
        layout.addRow(record_path_btn)

        save_btn = QPushButton("설정 저장")
        save_btn.clicked.connect(self.accept)
        layout.addRow(save_btn)
        self.setLayout(layout)

        for i, (type_combo, value_edit, resolution_combo, _) in enumerate(self.inputs):
            try:
                type_combo.setCurrentText(self.cam_sources[i].get("type", "USB"))
                value_edit.setText(self.cam_sources[i].get("value", str(i)))
                
                current_res = self.cam_sources[i].get("resolution", "1920x1080")
                if current_res in [resolution_combo.itemText(j) for j in range(resolution_combo.count())]:
                    resolution_combo.setCurrentText(current_res)
                else:
                    if resolution_combo.count() > 0:
                        resolution_combo.setCurrentText(resolution_combo.itemText(0))
                    else:
                        resolution_combo.setCurrentText("640x480") 
            except IndexError:
                type_combo.setCurrentText("USB")
                value_edit.setText(str(i))
                if resolution_combo.count() > 0:
                    resolution_combo.setCurrentText(resolution_combo.itemText(0))
                else:
                    resolution_combo.setCurrentText("640x480") 
        
        if self.cam_sources:
            self.record_path_edit.setText(self.cam_sources[0].get("record_dir", ""))
        else:
            self.record_path_edit.setText("")

    def select_record_dir(self):
        path = QFileDialog.getExistingDirectory(self, "녹화 폴더 선택")
        if path:
            self.record_path_edit.setText(path)

    def get_settings(self):
        result = []
        global_record_dir = self.record_path_edit.text().strip()
        for type_combo, value_edit, resolution_combo, _ in self.inputs:
            result.append({
                "type": type_combo.currentText(),
                "value": value_edit.text().strip(),
                "resolution": resolution_combo.currentText(),
                "record_dir": global_record_dir
            })
        return result

class MultiCamViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("logo.png"))
        self.setWindowTitle("AISEED 다중 카메라 스트리밍 뷰어")
        self.setGeometry(100, 100, 1920, 1080)

        self.labels = []
        self.streams = []
        self.expanded_label = None
        self.is_global_recording_active = False

        self.cam_sources = self.load_rtsp_config()
        self.camera_capabilities = [{'max_width': 0, 'max_height': 0, 'max_fps': 0.0} for _ in range(6)]


        self.start_streaming_button = QPushButton("스트리밍 시작")
        self.start_streaming_button.clicked.connect(self.start_streaming)

        self.stop_streaming_button = QPushButton("스트리밍 중지")
        self.stop_streaming_button.clicked.connect(self.stop_all_streams)
        self.stop_streaming_button.setEnabled(False)

        self.start_record_button = QPushButton("녹화 시작")
        self.start_record_button.clicked.connect(self.start_recording)
        self.start_record_button.setEnabled(False)

        self.stop_record_button = QPushButton("녹화 중지")
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)

        self.settings_button = QPushButton("설정")
        self.settings_button.clicked.connect(self.open_settings)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.settings_button)
        button_layout.addWidget(self.start_streaming_button)
        button_layout.addWidget(self.stop_streaming_button)
        button_layout.addWidget(self.start_record_button)
        button_layout.addWidget(self.stop_record_button)
        button_layout.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        self.grid_container = QWidget()
        self.grid_container.setLayout(self.grid_layout)
        self.grid_container.setStyleSheet("background-color: black;")

        for i in range(6):
            label = QLabel(f"Camera {i + 1}\n신호 없음")
            label.setStyleSheet("border: 0px solid black; background-color: black; color: white;")
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.mouseDoubleClickEvent = self.make_double_double_click_handler(label, i)
            self.labels.append(label)
            row, col = divmod(i, 3)
            self.grid_layout.addWidget(label, row, col, 1, 1)

        main_layout.addWidget(self.grid_container)
        self.setLayout(main_layout)

    def load_rtsp_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if len(config) == 6:
                        return config
                    else:
                        print("설정 파일의 항목 수가 올바르지 않습니다. 기본 설정을 사용합니다.")
            except Exception as e:
                print(f"설정 로드 실패: {e}")
        return [{"type": "USB", "value": str(i), "resolution": "1920x1080", "record_dir": ""} for i in range(6)]

    def save_rtsp_config(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cam_sources, f, indent=4, ensure_ascii=False)
            print("설정 저장 완료.")
        except Exception as e:
            print(f"설정 저장 실패: {e}")

    def update_camera_capabilities(self, cam_index, max_width, max_height, max_fps):
        if 0 <= cam_index < len(self.camera_capabilities):
            self.camera_capabilities[cam_index] = {
                'max_width': max_width,
                'max_height': max_height,
                'max_fps': max_fps
            }
            print(f"MultiCamViewer: CAM {cam_index + 1} 능력치 업데이트됨: {max_width}x{max_height} @ {max_fps:.2f}fps")

    def open_settings(self):
        if self.streams:
            for i, stream in enumerate(self.streams):
                if stream.is_alive() and stream.cap and stream.cap.isOpened():
                    self.update_camera_capabilities(i, stream.actual_max_width, stream.actual_max_height, stream.actual_max_fps)
                else:
                    self.update_camera_capabilities(i, 0, 0, 0.0) 

        dlg = SettingsDialog(self, self.cam_sources, self.camera_capabilities)
        if dlg.exec_():
            self.cam_sources = dlg.get_settings()
            self.save_rtsp_config()
            QMessageBox.information(self, "설정", "설정이 저장되었습니다. 스트리밍을 다시 시작하여 적용하세요.")

    def start_streaming(self):
        self.stop_all_streams()
        
        for i, label in enumerate(self.labels):
            label.clear()
            label.setText(f"Camera {i + 1}\n연결 중...")
            label.setStyleSheet("border: 0px solid black; background-color: black; color: white;")

        self.camera_capabilities = [{'max_width': 0, 'max_height': 0, 'max_fps': 0.0} for _ in range(6)]

        for i, cam in enumerate(self.cam_sources):
            cam_type = cam.get("type")
            val = cam.get("value")
            record_base_dir = cam.get("record_dir")
            resolution_str = cam.get("resolution", "1920x1080")

            stream = None
            if cam_type == "USB":
                try:
                    cam_index_val = int(val) if val else 0
                except ValueError:
                    self.labels[i].setText("유효하지 않은 USB 인덱스")
                    continue
                stream = CameraStream(camera_identifier=cam_index_val, label=self.labels[i],
                                      camera_index=i, record_base_dir=record_base_dir,
                                      resolution_str=resolution_str)
            elif cam_type == "RTSP":
                if not val:
                    self.labels[i].setText("RTSP URL 없음")
                    continue
                stream = CameraStream(camera_identifier=None, label=self.labels[i],
                                      camera_index=i, rtsp_url=val, record_base_dir=record_base_dir,
                                      resolution_str=resolution_str)
            else:
                self.labels[i].setText("유효하지 않은 카메라 유형")
                continue

            self.streams.append(stream)
            stream.start()

        time.sleep(2) 

        for i, stream in enumerate(self.streams):
            if stream.is_alive() and stream.cap and stream.cap.isOpened():
                self.update_camera_capabilities(i, stream.actual_max_width, stream.actual_max_height, stream.actual_max_fps)
            else:
                self.update_camera_capabilities(i, 0, 0, 0.0) 

        self.start_streaming_button.setEnabled(False)
        self.stop_streaming_button.setEnabled(True)
        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)

    def stop_all_streams(self):
        self.stop_recording()
        for stream in self.streams:
            stream.stop()
        self.streams = []

        for i, label in enumerate(self.labels):
            label.clear()
            label.setText(f"Camera {i + 1}\n신호 없음")
            label.setStyleSheet("border: 0px solid black; background-color: black; color: white;")
            if self.expanded_label: 
                self.make_double_double_click_handler(self.expanded_label, -1)(None) 

        self.start_streaming_button.setEnabled(True)
        self.stop_streaming_button.setEnabled(False)
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(False)
        print("모든 스트림이 중지되었습니다.")

    def start_recording(self):
        if not self.streams:
            QMessageBox.warning(self, "경고", "스트리밍을 먼저 시작해주세요!")
            return

        record_dir_set = False
        for cam_config in self.cam_sources:
            if cam_config.get("record_dir"):
                record_dir_set = True
                break
        
        if not record_dir_set:
            QMessageBox.warning(self, "경고", "설정에서 녹화 저장 경로를 먼저 설정해주세요!")
            return

        for stream in self.streams:
            stream.set_recording_active(True)
        self.is_global_recording_active = True
        
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        print("녹화 시작: 모든 활성 스트림에서 녹화가 시작되었습니다.")
        QMessageBox.information(self, "녹화 시작", "모든 활성 스트림에서 녹화가 시작되었습니다.")

    def stop_recording(self):
        for stream in self.streams:
            stream.set_recording_active(False)
        self.is_global_recording_active = False

        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        print("녹화 중지: 모든 활성 스트림에서 녹화가 중지되었습니다.")
        QMessageBox.information(self, "녹화 중지", "모든 활성 스트림에서 녹화가 중지되었습니다.")

    def make_double_double_click_handler(self, label, index):
        def handler(event):
            if self.expanded_label == label:
                for i, cam_label in enumerate(self.labels):
                    cam_label.clear() 
                    cam_label.setVisible(True)
                    cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    cam_label.setMinimumSize(1, 1)
                    cam_label.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)

                while self.grid_layout.count():
                    item = self.grid_layout.takeAt(0)
                    if item.widget():
                        item.widget().setParent(None)

                for i, cam_label in enumerate(self.labels):
                    row, col = divmod(i, 3)
                    self.grid_layout.addWidget(cam_label, row, col, 1, 1)

                self.grid_layout.update()
                self.grid_container.updateGeometry()
                self.expanded_label = None
                
                for i, stream in enumerate(self.streams):
                    self.labels[i].setText(f"Camera {i + 1}\n신호 연결됨")
                    if not stream.is_alive() or not stream.cap or not stream.cap.isOpened():
                         self.labels[i].setText(f"Camera {i + 1}\n신호 없음")

            else:
                for cam_label in self.labels:
                    self.grid_layout.removeWidget(cam_label)
                    cam_label.setVisible(False)
                
                label.setVisible(True)
                label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                label.setMinimumSize(1, 1)
                label.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
                self.grid_layout.addWidget(label, 0, 0, 2, 3) 
                self.grid_layout.update()
                self.grid_container.updateGeometry()
                self.expanded_label = label
                
                for stream in self.streams:
                    if stream.label == label and stream.label.pixmap():
                        stream.label.setPixmap(stream.label.pixmap())
                        break

        return handler

    def closeEvent(self, event):
        self.stop_all_streams()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("logo.png"))
    viewer = MultiCamViewer()
    viewer.show()
    sys.exit(app.exec_())