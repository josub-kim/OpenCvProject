import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageGrab
import time


# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)  # 웹캠을 사용하여 손 검출을 시도합니다.

# UI 관련 설정
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2

# 삼각형 좌표 설정
triangle_color = (0, 0, 255)  # 빨간색
triangle_points = [(200, 100), (100, 300), (300, 300)]  # 삼각형의 꼭짓점 좌표
triangle_center_x, triangle_center_y = 320, 300  # 삼각형의 중심 좌표 (분리)


#직각삼각형 죄표 설정
rightangle_color=(0,255,255)# 노란색
rightangle_points=[(0,200),(0,400),(200,400)]
rightangle_center_x,rightangle_center_y=200,250


#왼쪽상단 삼각형 좌표 설정
leftangle_color=(255,255,255) #흰색
leftangle_points=[(120,41),(120,241),(220,141)]
leftangle_center_x,leftangle_center_y=50,141



# 초기화된 변수 추가
thumb_x, thumb_y = 0, 0
parallelogram_x = 0  # 평행사변형 X 좌표 초기화
parallelogram_y = 0

# 사각형 초기 설정 (작은 네모)
small_rectangle_x, small_rectangle_y = 320, 240  # 초기 위치 (화면 중앙)
small_rectangle_width, small_rectangle_height = 101.5, 101.5
small_rectangle_color = (0, 255, 0)  # 초록색

# 중앙에 큰 네모
frame_height, frame_width = cap.get(4), cap.get(3)  # 프레임의 높이와 너비 가져오기
center_x, center_y = int(frame_width // 2), int(frame_height // 2)  # 중앙 좌표
large_rectangle_width, large_rectangle_height = 400, 400
large_rectangle_color = (255, 0, 0)  # 파란색


#고정삼각형 좌표 설정
fixangle_color=(255,255,0) #하늘색
fixangle_points=[(120,41),(520,41),(320,241)]
fixangle_center_x,fixangle_center_y=200,100




#평행사변형
parallelogram_color = (100, 100, 0)  # 파란색
parallelogram_base = 200  # 밑변 길이
parallelogram_height = 100  # 높이


#큰 삼각형
bigangle_color=(120,120,120) 
bigangle_points=[(520,41),(320,241),(520,441)]
bigangle_center_x,bigangle_center_y=520,141


cv2.namedWindow("Hand Detection with Shapes", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Detection with Shapes", 900, 650)
angle=45


#고정 변수
Rs=True
Ys=True
Gs=True
Ws=True
ph=True
gs=True

completed_coords = {
    "빨간 삼각형": (320, 293),
    "노란 삼각형": (321, 288),
    "평행사변형": (316, 340),
    "작은 사각형": (222, 241),
    "왼쪽 상단 삼각형": (201, 104),
    "큰 삼각형": (420, 241)
}

# 모든 도형이 완성 좌표 주변에 있는지 확인하는 함수
def check_shapes_completion(shape_coords, threshold=10):
    for shape_name, shape_coord in shape_coords.items():
        completed_coord = completed_coords.get(shape_name)
        if completed_coord is not None:
            distance = calculate_distance(shape_coord, completed_coord)
            if distance > threshold:
                return False
    return True

# 두 점 사이의 거리 계산 함수
def calculate_distance(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

timer_started = True  # 타이머 시작 상태로 설정
start_time = time.time()
end = True
elapsed_time =0


image=cv2.imread("project/explain.png",cv2.IMREAD_COLOR)
resized_img=cv2.resize(image,dsize=(800,600))
cv2.imshow("Explain",resized_img)


while True:

    # 프레임 읽기
    ret, frame = cap.read() 
    # BGR 이미지를 RGB 이미지로 변환
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

         

    # 화면에 경과 시간 표시
    if end==True:
        cv2.putText(frame, f"Time Elapsed: {elapsed_time:.1f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        

        
    if cv2.waitKey(1) == ord('e'):
        if Ws:
            print("흰삼 고정")
            Ws = False
        else:
            print("흰삼 고정해제")
            Ws = True

    if cv2.waitKey(1) == ord('f'):
        if gs:
            print("회삼 고정")
            gs = False
        else:
            print("회삼 고정해제")
            gs = True
    if cv2.waitKey(1) == ord('a'):
        if Rs:
            print("빨간삼긱형 고정")
            Rs = False
        else:
            print("빨간삼각형 고정해제")
            Rs = True
    if cv2.waitKey(1) == ord('b'):
        if Ys:
            print("노란삼각형 고정")
            Ys = False
        else:
            print("노란삼각형 고정해제")
            Ys = True

    if cv2.waitKey(1) == ord('c'):
        if Gs:
            print("초록 사각형 고정")
            Gs = False
        else:
            print("초록 사각형 고정해제")
            Gs = True
    if cv2.waitKey(1) == ord('d'):
        if ph:
            print("평행 사변형 고정")
            ph = False
        else:
            print("평행 사뱐향 고정해제")
            ph = True            


  

  # 모든 도형이 완성 좌표 주변에 있는지 확인
    success = check_shapes_completion({
        "빨간 삼각형": (triangle_center_x, triangle_center_y),
        "노란 삼각형": (rightangle_center_x, rightangle_center_y),
        "평행사변형": (parallelogram_x, parallelogram_y),
        "작은 사각형": (small_rectangle_x, small_rectangle_y),
        "왼쪽 상단 삼각형": (leftangle_center_x, leftangle_center_y),
        "큰 삼각형": (bigangle_center_x, bigangle_center_y)
    })

    
    if end == True:
        cv2.putText(frame, f"Time Elapsed: {elapsed_time:.1f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


     # 모든 도형이 완성되면 텍스트 표시
    if success and timer_started:
        cv2.putText(frame, f"CONGRATULATION ({elapsed_time:.1f} seconds)", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        cv2.putText(frame, f"Game End Press any", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        end=False

       

    # 손 검출 수행
    results = hands.process(frame_rgb)

    #양손조건   
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        left_landmarks = results.multi_hand_landmarks[0].landmark
        right_landmarks = results.multi_hand_landmarks[1].landmark
        left_thumb_x, left_thumb_y = int(left_landmarks[4].x * frame.shape[1]), int(left_landmarks[4].y * frame.shape[0])
        right_thumb_x, right_thumb_y = int(right_landmarks[4].x * frame.shape[1]), int(right_landmarks[4].y * frame.shape[0])
        left_littlefinger_x,left_littlefinger_y=int(left_landmarks[20].x*frame.shape[1]),int(left_landmarks[20].y*frame.shape[0])
        right_littlefinger_x,right_littlefinger_y=int(right_landmarks[20].x*frame.shape[1]),int(right_landmarks[20].y*frame.shape[0])
        if left_thumb_x < right_thumb_x and abs(left_thumb_y - right_thumb_y) < 20 and Ws:
            leftangle_center_x = left_thumb_x
            leftangle_center_y = left_thumb_y
        elif left_littlefinger_x<right_littlefinger_x and abs(left_littlefinger_y-right_littlefinger_y)<20 and gs:
            bigangle_center_x=left_littlefinger_x
            bigangle_center_y=left_littlefinger_y


    # 검출된 손 표시
    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            # 손 관절을 그리기 위한 빈 리스트 초기화
            landmark_points = []

            for point_idx, point in enumerate(landmarks.landmark):
                x, y = int(point.x * frame_width), int(point.y * frame_height)
                landmark_points.append((x, y))

                # 손 관절 점 그리기
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # 손 관절 사이에 선 그리기
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (5, 6), (6, 7), (7, 8), (0, 5),
                (9, 10), (10, 11), (11, 12), (0, 17),
                (13, 14), (14, 15), (15, 16),
                (17, 18), (18, 19), (19, 20)
            ]
            for connection in connections:
                cv2.line(frame, landmark_points[connection[0]], landmark_points[connection[1]], (0, 255, 0), 2)
                
                # 엄지
                thumb_x, thumb_y = landmark_points[4]
                #검지
                index_x, index_y = landmark_points[8]
                # 중지
                middle_x, middle_y = landmark_points[12]
                #약지
                ring_x, ring_y = landmark_points[16]    
                #새끼
                seggi_x,seggi_y=landmark_points[20]
        
            # 엄지와 검지가 완전히 닿았을 때 삼각형의 중심을 엄지 손가락 위치로 이동
        if np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2) < 8 and Rs:   
            triangle_center_x = int(thumb_x)
            triangle_center_y = int(thumb_y)


        # 엄지와 약지가 완전히 닿았을 때 작은 네모를 선택 또는 이동
        elif np.sqrt((thumb_x - ring_x)**2 + (thumb_y - ring_y)**2) < 8 and Gs:
            small_rectangle_x, small_rectangle_y = int(thumb_x), int(thumb_y)

            # 엄지와 중지가 완전히 닿았을 때 평행사변형 위치를 엄지 손가락 위치로 이동
        elif np.sqrt((thumb_x - middle_x)**2 + (thumb_y - middle_y)**2) < 8 and ph:
            parallelogram_x = int(thumb_x)
            parallelogram_y = int(thumb_y)

        #노란색 삼각형
        elif np.sqrt((thumb_x -seggi_x)**2+ (thumb_y-seggi_y)**2)<8 and Ys:
            rightangle_center_x=int(thumb_x)
            rightangle_center_y=int(thumb_y)


    if end==True:
        elapsed_time = time.time() - start_time
        #빨간삼각형 그리기
        triangle_base = 200
        triangle_height = 100
        triangle_points = [
        (triangle_center_x - triangle_base // 2, triangle_center_y + triangle_height // 2),
        (triangle_center_x + triangle_base // 2, triangle_center_y + triangle_height // 2),
        (triangle_center_x, triangle_center_y - triangle_height // 2)]
        triangle_pts = np.array(triangle_points, np.int32)
        triangle_pts = triangle_pts.reshape((-1, 1, 2))   #1행 2열
        cv2.polylines(frame, [triangle_pts], isClosed=True, color=triangle_color, thickness=2)
        cv2.fillPoly(frame, [np.array(triangle_points)], (0, 0, 255))



        #직각 삼각형 그리기
        rightangle_points=[(rightangle_center_x-200,rightangle_center_y-50),(rightangle_center_x-200,rightangle_center_y+150),(rightangle_center_x,rightangle_center_y+150)]
        rightangle_pts=np.array(rightangle_points,np.int32)
        rightangle_pts=rightangle_pts.reshape((-1,1,2))
        cv2.polylines(frame,[rightangle_pts],isClosed=True,color=rightangle_color,thickness=2)
        cv2.fillPoly(frame,[np.array(rightangle_points)],(0,255,255))

        #왼쪽 상단 삼각형 그리기
        leftangle_points=[(leftangle_center_x-80,leftangle_center_y-59),(leftangle_center_x-80,leftangle_center_y+141),(leftangle_center_x+20,leftangle_center_y+41)]
        leftangle_pts=np.array(leftangle_points,np.int32)
        leftangle_pts=leftangle_pts.reshape((-1,1,2))
        cv2.polylines(frame,[leftangle_pts],isClosed=True,color=leftangle_color,thickness=2)
        cv2.fillPoly(frame,[np.array(leftangle_points)],(255,255,255))


        #고정 삼각형 그리기
        fixangle_pts=np.array(fixangle_points,np.int32)
        fixangle_pts=fixangle_pts.reshape((-1,1,2))
        cv2.polylines(frame,[fixangle_pts],isClosed=True,color=fixangle_color,thickness=2)
        cv2.fillPoly(frame,[np.array(fixangle_points)],(255,255,0))


        #큰 삼각형 그리기
        bigangle_points=[(bigangle_center_x+100,bigangle_center_y-200),(bigangle_center_x-100,bigangle_center_y),(bigangle_center_x+100,bigangle_center_y+200)]
        bigangle_pts=np.array(bigangle_points,np.int32)
        bigangle_pts=bigangle_pts.reshape((-1,1,2))

        cv2.polylines(frame,[bigangle_pts],isClosed=True,color=bigangle_color,thickness=2)
        cv2.fillPoly(frame,[np.array(bigangle_points)],(120,120,120))

        # 작은 정사각형 그리기 (길이가 100루트 2)
        square_size = 100 * np.sqrt(2)  # 정사각형의 크기 설정
        half_square_size = square_size / 2  # 정사각형 반 너비

        square_points = [
            (small_rectangle_x - half_square_size, small_rectangle_y - half_square_size),
            (small_rectangle_x + half_square_size, small_rectangle_y - half_square_size),
            (small_rectangle_x + half_square_size, small_rectangle_y + half_square_size),
            (small_rectangle_x - half_square_size, small_rectangle_y + half_square_size)
        ]
        square_points = np.array(square_points, dtype=np.int32)
        rotation_matrix = cv2.getRotationMatrix2D((small_rectangle_x, small_rectangle_y), angle, 1)

        # 사각형 좌표를 회전 변환 적용
        rotated_square_points = cv2.transform(np.array([square_points]), rotation_matrix)[0]
        # 회전된 사각형 그리기
        cv2.fillPoly(frame, [rotated_square_points], small_rectangle_color)
        #중앙 큰네모
        cv2.rectangle(
            frame,
            (center_x - large_rectangle_width // 2, center_y - large_rectangle_height // 2),0
            (center_x + large_rectangle_width // 2, center_y + large_rectangle_height // 2),
            large_rectangle_color,
            0 
        )
        # 평행사변형의 밑변 길이와 윗변 길이 설정
        parallelogram_base = 200  # 밑변 길이
        parallelogram_upper_base = 200  # 윗변 길이
        parallelogram_height = 100  # 높이

        parallelogram_points = [
            (parallelogram_x - parallelogram_base // 2, parallelogram_y),#평행 사변형 밑변의 중심
            (parallelogram_x + parallelogram_base // 2, parallelogram_y),#평행 사변형의 밑변의 반대쪽 끝이라는데?
            (parallelogram_x + parallelogram_base // 2 + parallelogram_height, parallelogram_y + parallelogram_height), #윗변의 오른쪽 끝
            (parallelogram_x - parallelogram_base // 2 + parallelogram_height, parallelogram_y + parallelogram_height)
        ]

        # if(triangle_center_x)
        cv2.fillPoly(frame, [np.array(parallelogram_points)], parallelogram_color)
    # 화면에 출력
    cv2.imshow('Hand Detection with Shapes', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 


# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()