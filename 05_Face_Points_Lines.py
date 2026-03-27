from imutils import face_utils
import imutils
import dlib
import cv2
import os
import time
import numpy as np

#=================================================================
# 초기값 설정
#=================================================================
# 실행 중인 스크립트 파일의 디렉토리 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 디렉토리 경로 설정
Dataset_dir = os.path.abspath(os.path.join(script_dir, "../00_Dataset"))

# 랜드마크 모델 파일의 전체 경로
dataset_file = os.path.join(Dataset_dir, "shape_predictor_68_face_landmarks.dat")

# 모델 파일 존재 여부 확인
if not os.path.exists(dataset_file):
    print(f"오류: 랜드마크 모델 파일을 찾을 수 없습니다. 경로: {dataset_file}")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 에서 파일을 다운로드하여")
    print(f"'{Dataset_dir}' 폴더 안에 저장했는지 확인해주세요.")
    exit()

# 한글 경로 문제 해결: 임시 영문 경로로 복사
import tempfile
import shutil
temp_dir = tempfile.gettempdir()
temp_file = os.path.join(temp_dir, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(temp_file) or os.path.getmtime(dataset_file) > os.path.getmtime(temp_file):
    shutil.copy2(dataset_file, temp_file)
    print(f"모델 파일을 임시 경로로 복사했습니다: {temp_file}")

#=================================================================
# 얼굴 감지 및 랜드마크 처리
#=================================================================
# dlib 얼굴 감지기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(temp_file)

# 각 얼굴 부위의 인덱스를 가져옵니다.
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

time.sleep(2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("오류: 프레임을 읽을 수 없습니다.")
        break

    frame = imutils.resize(frame, width=720)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    rects = detector(gray, 1)

    # 감지된 얼굴에 대해 반복
    for rect in rects:
        # 랜드마크 예측 및 NumPy 배열로 변환
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 각 얼굴 부위의 좌표를 추출하여 선으로 그립니다.
        # cv2.polylines(이미지, [점들], 닫힘 여부, 색상, 두께)
        
        # 1. 턱선 (Jawline)
        jaw_pts = shape[jStart:jEnd]
        cv2.polylines(frame, [jaw_pts], isClosed=False, color=(255, 255, 0), thickness=1)

        # 2. 눈썹 (Eyebrows)
        right_eyebrow_pts = shape[rbStart:rbEnd]
        left_eyebrow_pts = shape[lbStart:lbEnd]
        cv2.polylines(frame, [right_eyebrow_pts], isClosed=False, color=(255, 255, 0), thickness=1)
        cv2.polylines(frame, [left_eyebrow_pts], isClosed=False, color=(255, 255, 0), thickness=1)

        # 3. 코 (Nose)
        nose_pts = shape[nStart:nEnd]
        cv2.polylines(frame, [nose_pts], isClosed=False, color=(255, 255, 0), thickness=1)

        # 4. 눈 (Eyes)
        right_eye_pts = shape[reStart:reEnd]
        left_eye_pts = shape[leStart:leEnd]
        cv2.polylines(frame, [right_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)
        cv2.polylines(frame, [left_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)

        # 5. 입 (Mouth)
        # 입은 바깥 입술과 안쪽 입술을 따로 그려야 자연스럽습니다.
        outer_mouth_pts = shape[mStart:60]  # 바깥 입술 인덱스: 48-59
        inner_mouth_pts = shape[60:mEnd]    # 안쪽 입술 인덱스: 60-67
        cv2.polylines(frame, [outer_mouth_pts], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [inner_mouth_pts], isClosed=True, color=(0, 255, 0), thickness=1)
        
    # 결과 화면 출력
    cv2.imshow("Facial Landmarks Outline", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
