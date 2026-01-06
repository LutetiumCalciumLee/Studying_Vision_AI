import cv2
import numpy as np
import os

# =================================================================
# 초기값 설정
# =================================================================
# 이미지 파일 경로 설정
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ 변수가 없는 환경(예: Jupyter Notebook)을 위한 처리
    script_dir = os.getcwd()

os.chdir(script_dir)
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))
image_path = os.path.join(Videos_dir, "input", "apple.png")

# =================================================================
# 이미지 읽기 (한글 경로 지원)
# =================================================================
# OpenCV는 Windows에서 한글 경로를 직접 처리하지 못하므로, 
# 바이너리로 읽어서 cv2.imdecode()로 디코딩
def imread_korean(path):
    """한글 경로를 포함한 이미지 파일을 읽는 함수"""
    with open(path, "rb") as f:
        image_bytes = f.read()
    numpy_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image

image = imread_korean(image_path)
if image is None:
    print(f"Error: 이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
    exit()

output_image = image.copy()

# =================================================================
# 1단계: HSV 색상 공간으로 변환
# =================================================================
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# =================================================================
# 2단계: '빨간색' 후보 영역 마스크 생성
# =================================================================
# 빨간색 범위 정의
lower_red1 = np.array([0, 100, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([165, 100, 80])
upper_red2 = np.array([180, 255, 255])

# 빨간색 마스크 생성 및 병합
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.add(mask1, mask2)

# =================================================================
# 3단계: '초록색' 영역 마스크 생성
# =================================================================
# 초록색 범위 정의 (일반적인 초록색 HSV 범위)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# 초록색 마스크 생성
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# =================================================================
# 4단계: 빨간색에서 초록색과 겹치는 부분 제거
# =================================================================
# 빨간색과 초록색이 겹치는 영역 찾기
overlapping_mask = cv2.bitwise_and(red_mask, green_mask)

# 빨간색 마스크에서 겹치는 영역을 제거하여 순수한 빨간색만 남기기
pure_red_mask = cv2.subtract(red_mask, overlapping_mask)

# 마스크의 노이즈 제거 및 경계 부드럽게 처리
pure_red_mask = cv2.GaussianBlur(pure_red_mask, (9, 9), 2, 2)

# =================================================================
# 5단계: 순수한 빨간색 영역에서만 '동그란' 모양 검출
# =================================================================
# HoughCircles를 순수한 빨간색 마스크에 적용
circles = cv2.HoughCircles(
    pure_red_mask,            # 입력 이미지: 초록색이 제거된 순수한 빨간색 마스크
    cv2.HOUGH_GRADIENT,
    dp=1.1,                   # 해상도 비율
    minDist=40,               # 원 중심 간의 최소 거리
    param1=50,                # Canny 엣지 상위 임계값
    param2=25,                # accumulator 임계값
    minRadius=15,             # 검출할 원의 최소 반지름
    maxRadius=80              # 검출할 원의 최대 반지름
)

# =================================================================
# 검출된 사과에 파란 테두리와 번호 표시 (수정된 부분)
# =================================================================
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for i, (x, y, r) in enumerate(circles, 1):  # enumerate로 1부터 시작하는 번호 생성
        # 1. 원본 이미지에 검출된 원 그리기 (파란색 테두리)
        cv2.circle(output_image, (x, y), r, (255, 0, 0), 3)
        
        # 2. 번호 텍스트 추가 (원과 같은 파란색으로 변경)
        # 텍스트 설정
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        
        # 텍스트 크기 계산 (중앙 정렬을 위해)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        
        # 번호 텍스트 그리기 (파란색 - 원과 같은 색상)
        cv2.putText(output_image, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)

    print(f"총 {len(circles)}개의 사과를 검출했습니다.")
else:
    print("사과를 찾을 수 없습니다.")

# =================================================================
# 결과 이미지 보여주기
# =================================================================
cv2.imshow("Original Image", image)
cv2.imshow("Red Mask (Before)", red_mask)          # 원래 빨간색 마스크
cv2.imshow("Green Mask", green_mask)               # 초록색 마스크
cv2.imshow("Pure Red Mask (After)", pure_red_mask) # 초록색 제거 후 순수한 빨간색 마스크
cv2.imshow("Detected Apples with Numbers", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
