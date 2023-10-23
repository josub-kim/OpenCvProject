import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)  # 웹캠을 사용하여 손 검출을 시도합니다.

# UI 관련 설정
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2
image=cv2.imread("project/explain.png",cv2.IMREAD_COLOR)
resized_img=cv2.resize(image,dsize=(800,800))
cv2.imshow("Explain",image)

# 모든 도형의 초기 설정
def init_shapes():
    global triangle_center_x, triangle_center_y
    global rightangle_center_x, rightangle_center_y
    global leftangle_center_x, leftangle_center_y
    global parallelogram_x, parallelogram_y
    global small_rectangle_x, small_rectangle_y
    global large_rectangle_width, large_rectangle_height
    global bigangle_center_x, bigangle_center_y
    
    triangle_center_x, triangle_center_y = 320, 300
    rightangle_center_x, rightangle_center_y = 200, 250
    leftangle_center_x, leftangle_center_y = 50, 141
    parallelogram_x, parallelogram_y = 0, 0
    small_rectangle_x, small_rectangle_y = 320, 240
    large_rectangle_width, large_rectangle_height = 400, 400
    bigangle_center_x, bigangle_center_y = 520, 141

# 모든 도형을 그리는 함수
def draw_shapes(frame):
    # 삼각형 그리기
    triangle_base = 200
    triangle_height = 100
    triangle_points = [
        (triangle_center_x - triangle_base // 2, triangle_center_y + triangle_height // 2),
        (triangle_center_x + triangle_base // 2, triangle_center_y + triangle_height // 2),
        (triangle_center_x, triangle_center_y - triangle_height // 2)]
    triangle_pts = np.array(triangle_points, np.int32)
    triangle_pts = triangle_pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [triangle_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.fillPoly(frame, [np.array(triangle_points)], (0, 0, 255))
    
    # 다른 도형들도 유사하게 그립니다.

# 모든 도형이 완성되었는지 확인하는 함수
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

# 키 입력을 처리하여 도형 고정/해제 상태를 변경하는 함수
def handle_fixed():
    global Rs, Ys, Gs, Ws, ph, gs
    
    key = cv2.waitKey(1)
    if key == ord('e'):
        Ws = not Ws
    elif key == ord('f'):
        gs = not gs
    elif key == ord('a'):
        Rs = not Rs
    elif key == ord('b'):
        Ys = not Ys
    elif key == ord('c'):
        Gs = not Gs
    elif key == ord('d'):
        ph = not ph

# 메인 루프
def main():
    global triangle_center_x, triangle_center_y
    global rightangle_center_x, rightangle_center_y
    global leftangle_center_x, leftangle_center_y
    global parallelogram_x, parallelogram_y
    global small_rectangle_x, small_rectangle_y
    global large_rectangle_width, large_rectangle_height
    global bigangle_center_x, bigangle_center_y
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_width, frame_height, _ = frame.shape

        handle_fixed()

        # 모든 도형이 완성 좌표 주변에 있는지 확인
        success = check_shapes_completion({
            "빨간 삼각형": (triangle_center_x, triangle_center_y),
            "노란 삼각형": (rightangle_center_x, rightangle_center_y),
            "평행사변형": (parallelogram_x, parallelogram_y),
            "작은 사각형": (small_rectangle_x, small_rectangle_y),
            "왼쪽 상단 삼각형": (leftangle_center_x, leftangle_center_y),
            "큰 삼각형": (bigangle_center_x, bigangle_center_y)
        })

        if success:
            cv2.putText(frame, "CONGRATULATION", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

        # 손 검출 수행
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            # 손 위치와 상태 업데이트
            # ...

            draw_shapes(frame)

        # 화면에 출력
        cv2.imshow('Hand Detection with Shapes', frame)
       

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 초기 설정
init_shapes()

# 고정 여부 설정
Rs, Ys, Gs, Ws, ph, gs = True, True, True, True, True, True

# 초기화된 변수 추가
completed_coords = {
    "빨간 삼각형": (320, 293),
    "노란 삼각형": (321, 288),
    "평행사변형": (316, 340),
    "작은 사각형": (222, 241),
    "왼쪽 상단 삼각형": (201, 104),
    "큰 삼각형": (420, 241)
}

# 메인 루프 실행
main()

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
