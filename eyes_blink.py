import cv2
import mediapipe as mp
import numpy as np
import time
import openai
import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()
# OpenAI GPT 초기화
openai.api_key = os.getenv('API_KEY')

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# TTS 초기화
engine = pyttsx3.init()

# EAR(눈 종횡비)를 계산하는 함수
def calculate_EAR(landmarks, left_indices, right_indices):
    left_eye = np.array([landmarks[i] for i in left_indices])
    right_eye = np.array([landmarks[i] for i in right_indices])

    left_EAR = (np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])) / (
                2 * np.linalg.norm(left_eye[0] - left_eye[3]))
    right_EAR = (np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])) / (
                2 * np.linalg.norm(right_eye[0] - right_eye[3]))

    return (left_EAR + right_EAR) / 2

# 눈의 랜드마크 인덱스
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# GPT와 대화하는 함수
def chat_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        return "요청 한도가 초과되었습니다. 잠시 후 다시 시도해 주세요."

# 음성을 텍스트로 변환하는 함수
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="ko-KR")
            return text
        except sr.UnknownValueError:
            return speak("이해하지 못했습니다.")
        except sr.RequestError:
            return speak("API를 사용할 수 없습니다.")

# 텍스트를 음성으로 변환하는 함수
def speak(text):
    engine.say(text)
    engine.runAndWait()

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

# EAR 임계값과 타이머 초기화
EAR_THRESHOLD = 0.25
TIME_THRESHOLD = 3  # 초
blink_count = 0
start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

            EAR = calculate_EAR(landmarks, left_eye_indices, right_eye_indices)

            if EAR < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()
                else:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= TIME_THRESHOLD:
                        blink_count += 1
                        start_time = None  # 타이머 초기화
                        # GPT와 대화 시작
                        gpt_response = chat_with_gpt("너는 졸음방지를 위한 대화 서비스야. 지금 운전자의 졸음을 감지했어 최대한 졸음을 깰 수 있도록 대화를 하게 될거야. '너는 사용자의 졸음이 감지되어 대화를 시도합니다' 라고 시작하며 사용자의 응답에 따라 이야기를 진행해")
                        print(f"GPT: {gpt_response}")
                        speak(gpt_response)
                        while True:
                            # 사용자의 응답 듣기
                            user_input = listen()
                            print(f"User: {user_input}")
                            if user_input.lower() == "종료":
                                speak("대화를 종료합니다.")
                                break
                            if user_input.lower() != "이해하지 못했습니다." and user_input.lower() != "API를 사용할 수 없습니다.":
                                gpt_response = chat_with_gpt(user_input)
                                print(f"GPT: {gpt_response}")
                                speak(gpt_response)
            else:
                start_time = None  # 타이머 초기화

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    cv2.putText(frame, f'Blinks: {blink_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Blink Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
