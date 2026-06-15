std_msgs/Header header
bool valid

float32 x_error        # m,  종방향 거리 오차 (pc_detector → manager → controller)
float32 y_error        # m,  횡방향 오차   (pc_detector, 참고용)
float32 theta_error    # rad, yaw 오차    (edge_detector → manager → controller)
float32 initial_dist_m # m,  시작 거리    (감속 ramp 산정용)
float32 mean_y_px      # px, 수평엣지 평균 y (edge_detector 출력)
