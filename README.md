<details> <summary>ENG (English Version)</summary>

# Drowsiness Detection Project

## Project Overview

This project is a real-time drowsiness detection system that monitors a driver's eyes through a webcam and alerts when drowsiness is detected. It uses facial landmark detection to calculate the Eye Aspect Ratio (EAR) and determines drowsiness based on eye closure duration. The system can detect even partially closed eyes (squinting) and provides visual warnings and beep alerts.

### Key Features
- Real-time face and eye detection using dlib 68-point facial landmarks
- Eye landmark visualization with polylines (eyes only, for drowsiness detection focus)
- Eye Aspect Ratio (EAR) calculation for accurate eye state detection
- Moving Average filtering to reduce false positives
- Detection of partially closed eyes (squinting)
- 1-second threshold for eye closure detection
- Visual warning display with Korean text support
- Beep alarm sound for immediate alert
- Thread-based alarm system with 30-second cooldown period
- Korean path support for Windows environment

## Technology Stack

| Technology | Version/Description |
|------|----------|
| **Python** | 3.12 |
| **OpenCV (cv2)** | Video capture and image processing |
| **dlib** | Face detection and 68-point landmark prediction |
| **imutils** | Image utilities (resize, transforms) |
| **NumPy** | Array operations and mathematical calculations |
| **PIL (Pillow)** | Image processing and Korean text rendering |
| **winsound** | Beep alarm sound generation |
| **threading** | Asynchronous alarm processing |
| **OS/Tempfile/Shutil** | Path handling and temp file management |

## Project Goals

1. Real-time detection of eye closure and drowsiness
2. Accurate detection including partially closed eyes (squinting)
3. Reduce false positives through Moving Average filtering
4. Provide immediate visual and audio alerts
5. Support Korean text display in Windows environment
6. Reliable model loading in Korean path environments

## Algorithm Explanation

### Overall Process Flow
```
Webcam frame
    ↓
Grayscale conversion
    ↓
dlib face detection
    ↓
68-point landmark prediction
    ↓
Extract left and right eye coordinates
    ↓
Calculate Eye Aspect Ratio (EAR) for each eye
    ↓
Calculate average EAR
    ↓
Apply Moving Average filter
    ↓
Check if EAR < threshold (0.30)
    ↓
Count consecutive frames below threshold
    ↓
If ≥ 30 frames (1 second): Display WARNING + Beep
    ↓
If ≥ 60 frames (2 seconds): Display drowsiness alert + Alarm
```

### Step 1: Model Preparation and Path Handling
- Locate `shape_predictor_68_face_landmarks.dat` using script path
- Copy model to temp directory to avoid Korean path issues
- Initialize face detector and landmark predictor

### Step 2: Video Input Setup
- Open webcam with `cv2.VideoCapture(0)`
- Resize frame to width 720 for performance
- Flip frame horizontally for mirror mode
- Wait 2 seconds for camera warm-up

### Step 3: Face Detection and Landmark Extraction
- Convert frame to grayscale
- Detect faces using `dlib.get_frontal_face_detector()`
- For each detected face, predict 68 landmark points
- Extract left and right eye coordinates using `face_utils.FACIAL_LANDMARKS_IDXS`
- Draw eye landmark outlines using `cv2.polylines()` for visual feedback:
  - **Eyes**: Cyan closed polylines (focus on eyes for drowsiness detection)

### Step 4: Eye Aspect Ratio (EAR) Calculation

The EAR is calculated using the following formula:

```python
def eye_aspect_ratio(eye):
    # Vertical distances
    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])
    # Horizontal distance
    c = euclidean_dist(eye[0], eye[3])
    ear = (a + b) / (1.5 * c)
    return ear
```

**EAR Formula Explanation:**
- **a, b**: Vertical distances between eye landmarks (top-bottom)
- **c**: Horizontal distance between eye corners (left-right)
- **EAR**: Ratio of vertical to horizontal distances
- When eyes are open: EAR is higher (typically > 0.30)
- When eyes are closed: EAR decreases significantly (< 0.30)

### Step 5: Moving Average Filtering

To reduce false positives from brief blinks:

```python
def calculate_average(value):
    g_data.append(value)
    if len(g_data) > g_window_Size:
        g_data = g_data[-g_window_Size:]
    if len(g_data) < g_window_Size:
        return 0.0
    return float(sum(g_data) / g_window_Size)
```

- Window size: 15 frames
- Smooths out rapid fluctuations
- Prevents false alarms from normal blinking

### Step 6: Drowsiness Detection Logic

1. **Eye Closure Detection (1 second)**
   - If `ear_avg < 0.30` for ≥ 30 consecutive frames (≈1 second)
   - Display "WARNING: 눈을 감았습니다!" message
   - Play beep sound (1000Hz, 200ms)
   - Mark eye landmarks with red dots

2. **Drowsiness Detection (2 seconds)**
   - If `ear_avg < 0.30` for ≥ 60 consecutive frames (≈2 seconds)
   - Display "졸음이 감지 되었습니다" message
   - Trigger alarm system (with 30-second cooldown)

### Step 7: Alarm System
- Thread-based alarm processing to avoid blocking main loop
- 30-second cooldown period between alarms
- Beep sound generation using `winsound.Beep()`

## Code Structure

### Main Functions

#### `calculate_average(value)`
Moving Average filter to smooth EAR values and reduce false positives.

**Parameters:**
- `value`: Current EAR value

**Returns:**
- Averaged EAR value (0.0 if insufficient data)

#### `euclidean_dist(ptA, ptB)`
Calculate Euclidean distance between two points.

**Parameters:**
- `ptA`, `ptB`: Point coordinates (NumPy arrays)

**Returns:**
- Euclidean distance

#### `eye_aspect_ratio(eye)`
Calculate Eye Aspect Ratio for a single eye.

**Parameters:**
- `eye`: Array of 6 eye landmark points

**Returns:**
- EAR value (float)

#### `alarm_notification()`
Generate beep alarm sound in a separate thread.

#### `start_Alarm()`
Manage alarm triggering with cooldown period (30 seconds).

### Code Sections

1. **Initial Setup** (Lines 18-30)
   - Script directory and path configuration
   - Model and video file paths
   - Dataset file location

2. **Moving Average Function** (Lines 33-46)
   - Sliding window average calculation
   - Window size: 15 frames

3. **EAR Calculation Functions** (Lines 48-62)
   - Euclidean distance calculation
   - Eye Aspect Ratio formula implementation

4. **Alarm Functions** (Lines 64-90)
   - Beep sound generation
   - Thread-based alarm management
   - Cooldown period handling

5. **Global Variables** (Lines 93-99)
   - Alarm timing variables
   - Moving Average window size and data
   - Blink counter

6. **Font Setup** (Lines 105-107)
   - Korean font loading for text rendering

7. **Model Loading** (Lines 109-127)
   - Korean path handling via temp file copy
   - Face detector and landmark predictor initialization

8. **Video Capture Setup** (Lines 133-139)
   - Webcam initialization
   - Error handling

9. **Main Processing Loop** (Lines 141-251)
   - Frame reading and preprocessing
   - Face detection
   - Landmark extraction and facial region visualization
   - EAR calculation and filtering
   - Drowsiness detection and alerting
   - Visual feedback rendering (polylines and warning messages)

## Execution Method

### 1. Environment Setup
```bash
conda activate vision_ai
pip install opencv-python dlib imutils numpy pillow
```

### 2. File Structure
```
Vision AI/
├── 00_Dataset/
│   └── shape_predictor_68_face_landmarks.dat
└── 04_Setting/
    └── 06.detect_drowsiness.py
```

### 3. Execution
```bash
python 04_Setting/06.detect_drowsiness.py
```

### 4. Usage
- Position face in front of webcam
- System will detect face and monitor eyes
- Close eyes for 1 second to see WARNING message
- Keep eyes closed for 2 seconds to trigger drowsiness alert
- Press `q` key to exit

## Performance and Results

### Detection Accuracy
- Successfully detects fully closed eyes
- Detects partially closed eyes (squinting) with EAR threshold 0.30
- Moving Average filter reduces false positives from normal blinking
- Real-time processing at ~30 FPS (720px width)

### Visualization Features
- **Eye Landmark Outlines**: Eye contours connected with cyan polylines (closed polylines)
  - Focus on eyes only for drowsiness detection
  - Visual feedback for eye state monitoring
- **Eye Landmark Points**: Green dots on key eye landmarks for EAR calculation
- **Warning Overlay**: Yellow background with red text when drowsiness detected

### Alert System
- **Visual Alert**: Yellow background with red text warning
- **Audio Alert**: 1000Hz beep sound (200ms duration)
- **Drowsiness Alert**: Red text message after 2 seconds
- **Alarm Cooldown**: 30 seconds between alarms

### Example Output
```
모델 파일을 임시 경로로 복사했습니다: C:\...\shape_predictor_68_face_landmarks.dat
Play beep
```

## Technical Features and Problems Solved

### 1. Korean Path Support
**Problem**: dlib model loading fails with Korean paths on Windows  
**Solution**: Copy model file to temp directory (`tempfile.gettempdir()`) with English path before loading

### 2. False Positive Reduction
**Problem**: Normal blinking triggers false alarms  
**Solution**: 
- Implement Moving Average filter (15-frame window)
- Require 1-second (30 frames) duration for eye closure detection
- Require 2-second (60 frames) duration for drowsiness detection

### 3. Squinting Detection
**Problem**: Partially closed eyes (squinting) not detected  
**Solution**: Lower EAR threshold to 0.30 (from typical 0.22-0.28) to detect squinting

### 4. Korean Text Rendering
**Problem**: OpenCV cannot render Korean text directly  
**Solution**: Use PIL (Pillow) with Korean font (`gulim.ttc`) to render text, then convert back to NumPy array

### 5. Non-blocking Alarm System
**Problem**: Alarm sound blocks main processing loop  
**Solution**: Use threading to run alarm in separate thread, allowing continuous video processing

### 6. Alarm Spam Prevention
**Problem**: Continuous alarms when eyes remain closed  
**Solution**: Implement 30-second cooldown period between alarms

## Possible Improvements

### 1. Head Pose Estimation
- Detect head position and angle
- Alert when driver looks away from road
- Improve detection accuracy with head orientation

### 2. Yawning Detection
- Add mouth landmark analysis
- Detect yawning as additional drowsiness indicator
- Combine with eye closure for more accurate detection

### 3. Machine Learning Enhancement
- Train custom model for better EAR threshold adaptation
- Personalize thresholds based on individual eye characteristics
- Use deep learning for more robust detection

### 4. Performance Optimization
- GPU acceleration for face detection
- Multi-threading for parallel processing
- ROI (Region of Interest) cropping for faster processing

### 5. Additional Features
- Record drowsiness events with timestamps
- Generate reports and statistics
- Integration with vehicle systems
- SMS notification (currently commented out)

### 6. Adaptive Thresholds
- Automatically adjust EAR threshold based on lighting conditions
- Personalize thresholds for different users
- Dynamic window size adjustment

## Learning Points

What was learned through this project:

1. **Facial Landmark Detection**
   - dlib 68-point facial landmark model usage
   - Eye region extraction and analysis
   - Coordinate system understanding

2. **Eye Aspect Ratio (EAR)**
   - Mathematical calculation of eye state
   - Threshold selection for detection
   - Handling individual differences

3. **Signal Processing**
   - Moving Average filtering for noise reduction
   - Frame-based temporal analysis
   - False positive reduction techniques

4. **Real-time Processing**
   - Video stream handling with OpenCV
   - Performance optimization techniques
   - Non-blocking alarm system design

5. **Image Processing**
   - Grayscale conversion
   - Frame resizing and flipping
   - Text rendering with PIL

6. **System Design**
   - Thread-based asynchronous processing
   - Cooldown mechanism implementation
   - Error handling and robustness

7. **Windows Environment Issues**
   - Korean path handling
   - Font management for Korean text
   - System sound API usage (winsound)

## Developer Information

**Project**: Vision AI Practice  
**Filename**: `06.detect_drowsiness.py`  
**Development Environment**: Windows, Python, OpenCV, dlib, NumPy, PIL, winsound  
**Purpose**: Real-time drowsiness detection for driver safety

</details>

<details> <summary>KOR (한국어 버전)</summary>

# 졸음 감지 프로젝트 (Drowsiness Detection Project)

## 프로젝트 개요

이 프로젝트는 웹캠을 통해 운전자의 눈을 실시간으로 모니터링하고 졸음이 감지되면 경고를 제공하는 졸음 감지 시스템입니다. 얼굴 랜드마크 감지를 사용하여 눈 종횡비(EAR)를 계산하고 눈을 감은 지속 시간을 기반으로 졸음을 판단합니다. 실눈 상태도 감지할 수 있으며 시각적 경고와 알림을 제공합니다.

### 주요 특징
- dlib 68점 얼굴 랜드마크를 이용한 실시간 얼굴 및 눈 감지
- 폴리라인을 이용한 눈 랜드마크 시각화 (졸음 감지에 필요한 눈 부분만)
- 눈 종횡비(EAR) 계산을 통한 정확한 눈 상태 감지
- 오탐 방지를 위한 Moving Average 필터링
- 실눈(반쯤 감은 눈) 상태 감지
- 1초 이상 눈을 감으면 감지하는 임계값
- 한글 텍스트 지원 시각적 경고 표시
- 즉각적인 알림음
- 30초 쿨다운 기간을 가진 스레드 기반 알람 시스템
- Windows 환경에서의 한글 경로 지원

## 기술 스택

| 기술 | 버전/설명 |
|------|----------|
| **Python** | 3.12 |
| **OpenCV (cv2)** | 영상 캡처 및 이미지 처리 |
| **dlib** | 얼굴 감지 및 68점 랜드마크 예측 |
| **imutils** | 이미지 유틸리티(리사이즈, 변환 등) |
| **NumPy** | 배열 연산 및 수학 계산 |
| **PIL (Pillow)** | 이미지 처리 및 한글 텍스트 렌더링 |
| **winsound** | 알림음 생성 |
| **threading** | 비동기 알람 처리 |
| **OS/Tempfile/Shutil** | 파일 경로 및 임시 파일 관리 |

## 프로젝트 목표

1. 눈 감음 및 졸음의 실시간 감지
2. 실눈(반쯤 감은 눈) 상태까지 정확히 감지
3. Moving Average 필터링을 통한 오탐 감소
4. 즉각적인 시각 및 음향 알림 제공
5. Windows 환경에서 한글 텍스트 표시 지원
6. 한글 경로 환경에서 안정적인 모델 로딩

## 알고리즘 설명

### 전체 프로세스 흐름
```
웹캠 프레임
    ↓
그레이스케일 변환
    ↓
dlib 얼굴 감지
    ↓
68점 랜드마크 예측
    ↓
왼쪽 및 오른쪽 눈 좌표 추출
    ↓
각 눈의 눈 종횡비(EAR) 계산
    ↓
평균 EAR 계산
    ↓
Moving Average 필터 적용
    ↓
EAR < 임계값(0.30) 확인
    ↓
임계값 미만 연속 프레임 카운트
    ↓
≥ 30프레임(1초): WARNING 표시 + 알림음
    ↓
≥ 60프레임(2초): 졸음 경고 표시 + 알람
```

### 1단계: 모델 준비 및 경로 처리
- 스크립트 경로 기반으로 `shape_predictor_68_face_landmarks.dat` 위치 확인
- 한글 경로 문제를 피하기 위해 임시 디렉터리로 모델 복사
- 얼굴 감지기 및 랜드마크 예측기 초기화

### 2단계: 영상 입력 설정
- `cv2.VideoCapture(0)`으로 웹캠 열기
- 성능을 위해 프레임을 폭 720으로 리사이즈
- 거울 모드를 위해 프레임 좌우 반전
- 카메라 예열을 위해 2초 대기

### 3단계: 얼굴 감지 및 랜드마크 추출
- 프레임을 그레이스케일로 변환
- `dlib.get_frontal_face_detector()`로 얼굴 검출
- 검출된 얼굴마다 68점 랜드마크 예측
- `face_utils.FACIAL_LANDMARKS_IDXS`를 사용하여 왼쪽 및 오른쪽 눈 좌표 추출
- 시각적 피드백을 위해 `cv2.polylines()`로 눈 랜드마크 윤곽선 그리기:
  - **눈**: 청록색 닫힌 폴리라인 (졸음 감지에 필요한 눈 부분만 시각화)

### 4단계: 눈 종횡비(EAR) 계산

다음 공식을 사용하여 EAR를 계산합니다:

```python
def eye_aspect_ratio(eye):
    # 세로 거리
    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])
    # 가로 거리
    c = euclidean_dist(eye[0], eye[3])
    ear = (a + b) / (1.5 * c)
    return ear
```

**EAR 공식 설명:**
- **a, b**: 눈 랜드마크 간의 세로 거리(위-아래)
- **c**: 눈 모서리 간의 가로 거리(좌-우)
- **EAR**: 세로와 가로 거리의 비율
- 눈이 열려 있을 때: EAR가 높음 (일반적으로 > 0.30)
- 눈이 감혀 있을 때: EAR가 크게 감소 (< 0.30)

### 5단계: Moving Average 필터링

일반적인 깜빡임으로 인한 오탐을 줄이기 위해:

```python
def calculate_average(value):
    g_data.append(value)
    if len(g_data) > g_window_Size:
        g_data = g_data[-g_window_Size:]
    if len(g_data) < g_window_Size:
        return 0.0
    return float(sum(g_data) / g_window_Size)
```

- 윈도우 크기: 15프레임
- 급격한 변동을 평활화
- 일반적인 깜빡임으로 인한 오경보 방지

### 6단계: 졸음 감지 로직

1. **눈 감음 감지 (1초)**
   - `ear_avg < 0.30`이 30프레임 이상(≈1초) 지속되면
   - "WARNING: 눈을 감았습니다!" 메시지 표시
   - 알림음 재생 (1000Hz, 200ms)
   - 눈 랜드마크에 빨간 점 표시

2. **졸음 감지 (2초)**
   - `ear_avg < 0.30`이 60프레임 이상(≈2초) 지속되면
   - "졸음이 감지 되었습니다" 메시지 표시
   - 알람 시스템 트리거 (30초 쿨다운)

### 7단계: 알람 시스템
- 메인 루프를 차단하지 않기 위한 스레드 기반 알람 처리
- 알람 간 30초 쿨다운 기간
- `winsound.Beep()`을 사용한 알림음 생성

## 코드 구조

### 주요 함수

#### `calculate_average(value)`
EAR 값을 평활화하고 오탐을 줄이기 위한 Moving Average 필터.

**매개변수:**
- `value`: 현재 EAR 값

**반환값:**
- 평균화된 EAR 값 (데이터 부족 시 0.0)

#### `euclidean_dist(ptA, ptB)`
두 점 간의 유클리드 거리를 계산.

**매개변수:**
- `ptA`, `ptB`: 점 좌표 (NumPy 배열)

**반환값:**
- 유클리드 거리

#### `eye_aspect_ratio(eye)`
단일 눈의 눈 종횡비를 계산.

**매개변수:**
- `eye`: 6개의 눈 랜드마크 점 배열

**반환값:**
- EAR 값 (float)

#### `alarm_notification()`
별도 스레드에서 알림음 생성.

#### `start_Alarm()`
쿨다운 기간(30초)을 가진 알람 트리거 관리.

### 코드 섹션

1. **초기값 설정** (라인 18-30)
   - 스크립트 디렉터리 및 경로 구성
   - 모델 및 영상 파일 경로
   - 데이터셋 파일 위치

2. **Moving Average 함수** (라인 33-46)
   - 슬라이딩 윈도우 평균 계산
   - 윈도우 크기: 15프레임

3. **EAR 계산 함수** (라인 48-62)
   - 유클리드 거리 계산
   - 눈 종횡비 공식 구현

4. **알람 함수** (라인 64-90)
   - 알림음 생성
   - 스레드 기반 알람 관리
   - 쿨다운 기간 처리

5. **전역 변수** (라인 93-99)
   - 알람 타이밍 변수
   - Moving Average 윈도우 크기 및 데이터
   - 깜빡임 카운터

6. **폰트 설정** (라인 105-107)
   - 텍스트 렌더링을 위한 한글 폰트 로딩

7. **모델 로딩** (라인 109-127)
   - 임시 파일 복사를 통한 한글 경로 처리
   - 얼굴 감지기 및 랜드마크 예측기 초기화

8. **영상 캡처 설정** (라인 133-139)
   - 웹캠 초기화
   - 에러 처리

9. **메인 처리 루프** (라인 141-251)
   - 프레임 읽기 및 전처리
   - 얼굴 감지
   - 랜드마크 추출 및 얼굴 부위 시각화
   - EAR 계산 및 필터링
   - 졸음 감지 및 알림
   - 시각적 피드백 렌더링 (폴리라인 및 경고 메시지)

## 실행 방법

### 1. 환경 설정
```bash
conda activate vision_ai
pip install opencv-python dlib imutils numpy pillow
```

### 2. 파일 구조
```
Vision AI/
├── 00_Dataset/
│   └── shape_predictor_68_face_landmarks.dat
└── 04_Setting/
    └── 06.detect_drowsiness.py
```

### 3. 실행
```bash
python 04_Setting/06.detect_drowsiness.py
```

### 4. 사용 방법
- 웹캠 앞에 얼굴을 위치시킵니다
- 시스템이 얼굴을 감지하고 눈을 모니터링합니다
- 1초 동안 눈을 감으면 WARNING 메시지가 표시됩니다
- 2초 동안 눈을 감으면 졸음 경고가 트리거됩니다
- `q` 키를 눌러 종료합니다

## 성능 및 결과

### 검출 정확도
- 완전히 감은 눈을 성공적으로 감지
- EAR 임계값 0.30으로 실눈(반쯤 감은 눈) 상태 감지
- Moving Average 필터가 일반적인 깜빡임으로 인한 오탐 감소
- 약 30 FPS로 실시간 처리 (720px 폭)

### 시각화 기능
- **눈 랜드마크 윤곽선**: 눈 윤곽선을 청록색 폴리라인으로 연결하여 표시 (닫힌 폴리라인)
  - 졸음 감지에 필요한 눈 부분만 시각화
  - 눈 상태 모니터링을 위한 시각적 피드백
- **눈 랜드마크 점**: EAR 계산을 위한 주요 눈 랜드마크에 녹색 점 표시
- **경고 오버레이**: 졸음 감지 시 빨간색 텍스트가 있는 노란색 배경

### 알림 시스템
- **시각적 알림**: 빨간색 텍스트가 있는 노란색 배경 경고
- **음향 알림**: 1000Hz 알림음 (200ms 지속 시간)
- **졸음 경고**: 2초 후 빨간색 텍스트 메시지
- **알람 쿨다운**: 알람 간 30초

### 예시 출력
```
모델 파일을 임시 경로로 복사했습니다: C:\...\shape_predictor_68_face_landmarks.dat
Play beep
```

## 기술적 특징 및 해결한 문제

### 1. 한글 경로 지원
**문제**: Windows에서 dlib 모델 로딩이 한글 경로에서 실패  
**해결**: 로딩 전에 임시 디렉터리(`tempfile.gettempdir()`)에 영문 경로로 모델 파일 복사

### 2. 오탐 감소
**문제**: 일반적인 깜빡임이 오경보를 트리거  
**해결**: 
- Moving Average 필터 구현 (15프레임 윈도우)
- 눈 감음 감지를 위해 1초(30프레임) 지속 시간 요구
- 졸음 감지를 위해 2초(60프레임) 지속 시간 요구

### 3. 실눈 감지
**문제**: 반쯤 감은 눈(실눈) 상태가 감지되지 않음  
**해결**: 실눈을 감지하기 위해 EAR 임계값을 0.30으로 낮춤 (일반적인 0.22-0.28에서)

### 4. 한글 텍스트 렌더링
**문제**: OpenCV가 한글 텍스트를 직접 렌더링할 수 없음  
**해결**: 한글 폰트(`gulim.ttc`)를 사용하여 PIL(Pillow)로 텍스트를 렌더링한 후 NumPy 배열로 다시 변환

### 5. 논블로킹 알람 시스템
**문제**: 알람 소리가 메인 처리 루프를 차단  
**해결**: 스레딩을 사용하여 별도 스레드에서 알람을 실행하여 연속적인 영상 처리를 허용

### 6. 알람 스팸 방지
**문제**: 눈이 계속 감혀 있을 때 연속적인 알람 발생  
**해결**: 알람 간 30초 쿨다운 기간 구현

## 개선 가능한 사항

### 1. 머리 자세 추정
- 머리 위치 및 각도 감지
- 운전자가 도로에서 시선을 돌릴 때 경고
- 머리 방향으로 검출 정확도 향상

### 2. 하품 감지
- 입 랜드마크 분석 추가
- 추가 졸음 지표로 하품 감지
- 눈 감음과 결합하여 더 정확한 감지

### 3. 머신러닝 향상
- 더 나은 EAR 임계값 적응을 위한 커스텀 모델 학습
- 개인 눈 특성에 따른 임계값 개인화
- 더 강력한 감지를 위한 딥러닝 사용

### 4. 성능 최적화
- 얼굴 감지를 위한 GPU 가속
- 병렬 처리를 위한 멀티스레딩
- 더 빠른 처리를 위한 ROI(관심 영역) 크롭

### 5. 추가 기능
- 타임스탬프가 있는 졸음 이벤트 기록
- 보고서 및 통계 생성
- 차량 시스템과의 통합
- SMS 알림 (현재 주석 처리됨)

### 6. 적응형 임계값
- 조명 조건에 따라 EAR 임계값 자동 조정
- 다른 사용자에 대한 임계값 개인화
- 동적 윈도우 크기 조정

## 학습 포인트

이 프로젝트를 통해 학습한 내용:

1. **얼굴 랜드마크 감지**
   - dlib 68점 얼굴 랜드마크 모델 사용
   - 눈 영역 추출 및 분석
   - 좌표계 이해

2. **눈 종횡비(EAR)**
   - 눈 상태의 수학적 계산
   - 감지를 위한 임계값 선택
   - 개인차 처리

3. **신호 처리**
   - 노이즈 감소를 위한 Moving Average 필터링
   - 프레임 기반 시간 분석
   - 오탐 감소 기법

4. **실시간 처리**
   - OpenCV를 사용한 영상 스트림 처리
   - 성능 최적화 기법
   - 논블로킹 알람 시스템 설계

5. **이미지 처리**
   - 그레이스케일 변환
   - 프레임 리사이즈 및 반전
   - PIL을 사용한 텍스트 렌더링

6. **시스템 설계**
   - 스레드 기반 비동기 처리
   - 쿨다운 메커니즘 구현
   - 에러 처리 및 견고성

7. **Windows 환경 문제**
   - 한글 경로 처리
   - 한글 텍스트를 위한 폰트 관리
   - 시스템 사운드 API 사용 (winsound)

## 개발자 정보

**프로젝트**: Vision AI 실습  
**파일명**: `06.detect_drowsiness.py`  
**개발 환경**: Windows, Python, OpenCV, dlib, NumPy, PIL, winsound  
**목적**: 운전자 안전을 위한 실시간 졸음 감지

</details>
