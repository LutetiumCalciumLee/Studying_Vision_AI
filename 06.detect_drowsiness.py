from imutils import face_utils
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from PIL import Image, ImageFont, ImageDraw
import pygame
import threading
import tempfile
import shutil
import winsound
#from class_naver_sms_service import naver_sms_sender

#=================================================================
#초기값 설정
#=================================================================
#실행 경로 설정 
# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))
Dataset_dir = os.path.abspath(os.path.join(script_dir, "../00_Dataset"))
video_file = os.path.join(Videos_dir, "input","driver.mp4" )
alarm_file = os.path.join(Videos_dir, "input","alarm.wav" )
dataset_file = os.path.join(Dataset_dir, "shape_predictor_68_face_landmarks.dat" )


#=================================================================
#눈동자 가로,세로 비율의 Moving Average 처리
#=================================================================
def calculate_average(value):
	global g_window_Size
	global g_data

	g_data.append(value)
	if len(g_data) > g_window_Size:
		g_data = g_data[-g_window_Size:]
	
	if len(g_data) < g_window_Size:
		return 0.0
	return float(sum(g_data) / g_window_Size)

#=================================================================
# 눈동자 가로, 세로euclidean거리 구하기
#=================================================================
def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)

#눈의 가로, 세로 종횡비 구하기 
def eye_aspect_ratio(eye):
	#눈의 세로 
	a = euclidean_dist(eye[1], eye[5])
	b = euclidean_dist(eye[2], eye[4])
	#눈의 가로 
	c = euclidean_dist(eye[0], eye[3])
	ear = (a + b) / (1.5 * c)
	return ear

#=================================================================
#졸음 감지 시 알림 처리 beep 경고음
#=================================================================
def alarm_notification():
	'''
	print("Send SMS")
	sms_sender=naver_sms_sender()
	sms_sender.send_sms(to_number)
	'''
	print("Play beep")
	# 1000Hz 주파수로 500ms 동안 beep 소리 재생
	winsound.Beep(1000, 500)

#=================================================================
#Alarm 처리 Thread 
#마지막 알람 발생 후 30초 후 알람 발생 
#=================================================================
def start_Alarm():
	global g_pre_alarm_time
	cur_time = time.time()
	
	if (cur_time - g_pre_alarm_time) > 30:
		thread = threading.Thread(target=alarm_notification)
		thread.start()
		g_pre_alarm_time = cur_time
	else:
		print("Alarm is not progress time: {0}s.".format(int(cur_time - g_pre_alarm_time)))


#=================================================================
#초기값 설정
#=================================================================
g_pre_alarm_time = 0
g_window_Size = 15  # Moving Average 윈도우 크기 감소 (더 빠른 반응)
g_data =[]
g_blinkCounter = 0

#실행 경로 설정 
this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)

#한글 폰트 설정 
fontpath = r"C:\Windows\Fonts\gulim.ttc"  # raw string으로 변경하여 SyntaxWarning 해결
font = ImageFont.truetype(fontpath, 36)

#=================================================================
#얼굴 감지, 눈동자 감지 처리 
#=================================================================
# 한글 경로 문제 해결: 임시 영문 경로로 복사
if not os.path.exists(dataset_file):
    print(f"오류: 랜드마크 모델 파일을 찾을 수 없습니다. 경로: {dataset_file}")
    exit()

temp_dir = tempfile.gettempdir()
temp_file = os.path.join(temp_dir, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(temp_file) or os.path.getmtime(dataset_file) > os.path.getmtime(temp_file):
    shutil.copy2(dataset_file, temp_file)
    print(f"모델 파일을 임시 경로로 복사했습니다: {temp_file}")

#face detecor pre trained NN 
detector = dlib.get_frontal_face_detector()

#dlib's facial landmark NN 초기화
predictor = dlib.shape_predictor(temp_file)

# 오른쪽, 왼쪽 눈 좌표 인덱스 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
	print("Error: 비디오 파일을 열 수 없습니다.")
	exit()
time.sleep(2.0)

while True:
	#웹캠 영상 읽기
	ret, frame = cap.read()
	if not ret:
		print("비디오가 끝났거나 에러가 발생했습니다.")
		break

	frame = imutils.resize(frame, width=720)
	frame = cv2.flip(frame, 1)

	#입력영상 graysale 처리 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# 얼굴 Detection 
	rects = detector(gray)	
   
	for rect in rects:
		x, y = rect.left(), rect.top()
		w, h = rect.right() - x, rect.bottom() - y
		
		#print( w, h)
		#얼굴 크기가 110이상일때만 눈동자 Detection
		#if (w > 110 ):
		#얼굴 영역 bounding box 그리기
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

		#눈동자 Detection(68 landmarks)
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		#왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 양쪽 눈의 눈 종횡비를 계산
		# 눈 부분만 라인으로 연결 (졸음 감지에 필요한 부분만 시각화)
		right_eye_pts = shape[rStart:rEnd]
		left_eye_pts = shape[lStart:lEnd]
		cv2.polylines(frame, [right_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)
		cv2.polylines(frame, [left_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)
		
		#왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 양쪽 눈의 눈 종횡비를 계산
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		
		#print( ear_avg )
		# 양쪽 눈동자 외각선 찾기
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		#양쪽 눈동자 녹색 외각선 그리기
		#cv2.drawContours(frame, [leftEyeHull], 0, (0, 255, 0), 1)
		#cv2.drawContours(frame, [rightEyeHull], 0, (0, 255, 0), 1)
		#양쪽 눈동자 녹색 점 그리기
		cv2.circle(frame, tuple(rightEye[[0,0][0]]), 2, (0,255,0), -1)
		cv2.circle(frame, tuple(rightEye[[1,0][0]]), 2, (0,255,0), -1)
		cv2.circle(frame, tuple(rightEye[[2,0][0]]), 2, (0,0,0), -1)
		cv2.circle(frame, tuple(rightEye[[3,0][0]]), 2, (0,0,0), -1)
		cv2.circle(frame, tuple(rightEye[[4,0][0]]), 2, (0,0,0), -1)
		cv2.circle(frame, tuple(rightEye[[5,0][0]]), 2, (0,0,0), -1)   
		
		cv2.circle(frame, tuple(leftEye[[0,0][0]]), 2, (0,255,0), -1)
		cv2.circle(frame, tuple(leftEye[[1,0][0]]), 2, (0,255,0), -1)
		cv2.circle(frame, tuple(leftEye[[2,0][0]]), 2, (0,0,0), -1)
		cv2.circle(frame, tuple(leftEye[[3,0][0]]), 2, (0,0,0), -1)
		cv2.circle(frame, tuple(leftEye[[4,0][0]]), 2, (0,0,0), -1)
		cv2.circle(frame, tuple(leftEye[[5,0][0]]), 2, (0,0,0), -1)

		#print( leftEAR, rightEAR )
		#양쪽 눈의 종횡비 평균
		ear = (leftEAR + rightEAR) / 2.0
		#양쪽 눈의 종횡비율의 Moving Average 처리(오탐 방지)
		ear_avg= calculate_average(ear)

		# 눈 가로 세로 비율이 0.30 미만이면 실눈/눈감음으로 판단 (실눈도 감지)
		if ear_avg < 0.30:
			# 깜박임 회수 + 1
			g_blinkCounter += 1
			
			# 1초 이상(약 30프레임) 눈을 감았으면 눈을 감은 것으로 인식
			if g_blinkCounter >= 30:
				# 눈을 감은 것이 감지되면 WARNING 표시
				img_pillow = Image.fromarray(frame)
				draw = ImageDraw.Draw(img_pillow, 'RGBA')
				# WARNING 텍스트 표시 (노란색 배경에 빨간색 텍스트)
				warning_text = "WARNING: 눈을 감았습니다!"
				text_bbox = draw.textbbox((0, 0), warning_text, font=font)
				text_width = text_bbox[2] - text_bbox[0]
				text_height = text_bbox[3] - text_bbox[1]
				# 배경 박스 그리기
				draw.rectangle([(5, 5), (5 + text_width + 10, 5 + text_height + 10)], fill=(0, 255, 255, 200))
				# 경고 텍스트 그리기
				draw.text((10, 10), warning_text, (0, 0, 255), font=font)
				frame = np.array(img_pillow)
				
				# 양쪽 눈동자 빨간 점 그리기
				cv2.circle(frame, tuple(rightEye[[3,0][0]]), 3, (0,0,255), -1)
				cv2.circle(frame, tuple(leftEye[[0,0][0]]), 3, (0,0,255), -1)
				
				# beep 경고음 재생
				winsound.Beep(1000, 200)

			# 깜박임 회수가 60회(약 2초) 이상이면 졸음으로 판단
			if g_blinkCounter >= 60:				
				img_pillow = Image.fromarray(frame)
				draw = ImageDraw.Draw(img_pillow, 'RGBA')
				draw.text((5, 60), "졸음이 감지 되었습니다", (0,0,255), font=font)
				frame = np.array( img_pillow )
				start_Alarm()
		else:
			g_blinkCounter = 0
				
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()