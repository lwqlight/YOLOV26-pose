import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter

# ================= 配置参数 =================
CONF_THRESHOLD = 0.5       # 目标检测置信度阈值
KPT_CONF_THRESHOLD = 0.5   # 关键点可见性阈值
SMOOTH_FRAMES = 10         # 平滑缓冲帧数 (数值越大越稳定，但延迟越高)

class HomePoseMonitor:
    def __init__(self):
        # 定义关键点索引 (COCO)
        self.KP = {
            'nose': 0, 
            'shoulders': [5, 6],
            'hips': [11, 12],
            'knees': [13, 14],
            'ankles': [15, 16]
        }
        # 状态缓冲区：{track_id: deque([pose1, pose2...])}
        self.pose_history = {}

    def is_valid_target(self, kpts, confs):
        """
        过滤逻辑：
        1. 必须能看到至少一个肩膀和一个胯部（躯干完整）
        2. 关键点整体平均置信度不能太低
        """
        if confs is None: return True
        
        # 检查躯干关键点 (肩膀5,6 胯部11,12)
        torso_indices = self.KP['shoulders'] + self.KP['hips']
        visible_torso_cnt = sum(1 for i in torso_indices if confs[i] > KPT_CONF_THRESHOLD)
        
        # 如果躯干关键点少于3个，认为遮挡严重或检测错误，过滤掉
        if visible_torso_cnt < 3:
            return False
        return True

    def calculate_angle(self, p1, p2, p3):
        """计算 p1-p2-p3 的夹角"""
        v1 = p1 - p2
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6: return 0
        cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle

    def classify_pose(self, kpts, box):
        """
        姿态核心算法
        kpts: (17, 2) 坐标
        box: (x1, y1, x2, y2) 检测框
        """
        # 1. 提取坐标
        l_shoulder, r_shoulder = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        l_knee, r_knee = kpts[13], kpts[14]
        l_ankle, r_ankle = kpts[15], kpts[16]

        # 计算身体中心点
        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2

        # --- A. “躺下”判定 (增强版) ---
        # 策略1：躯干倾斜角 (适用于侧身躺)
        dy = abs(hip_mid[1] - shoulder_mid[1])
        dx = abs(hip_mid[0] - shoulder_mid[0])
        inclination = np.degrees(np.arctan2(dx, dy)) if dy != 0 else 90

        # 策略2：检测框宽高比 (适用于正对镜头躺)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / h

        # 判定：如果身体很平(>65度) 或者 框很扁(宽是高的1.2倍以上)
        if inclination > 60 or aspect_ratio > 1.2:
            return "Lying Down"

        # --- B. “坐/站”判定 ---
        # 计算膝盖角度
        angle_l = self.calculate_angle(l_hip, l_knee, l_ankle)
        angle_r = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        # 取较大的角度（直的那条腿决定站立）
        # 如果有一条腿置信度低（比如坐标是0,0），只取另一条
        valid_angles = []
        if l_knee[0] > 0: valid_angles.append(angle_l)
        if r_knee[0] > 0: valid_angles.append(angle_r)
        
        if not valid_angles: return "Unknown"
        max_knee_angle = max(valid_angles)

        if max_knee_angle > 140: # 稍微放宽站立标准
            return "Standing"
        else:
            return "Sitting"

    def smooth_prediction(self, track_id, current_pose):
        """
        防抖动逻辑：滑动窗口投票
        """
        if track_id not in self.pose_history:
            self.pose_history[track_id] = deque(maxlen=SMOOTH_FRAMES)
        
        # 加入当前帧结果
        self.pose_history[track_id].append(current_pose)
        
        # 投票：取出现次数最多的姿态
        counts = Counter(self.pose_history[track_id])
        most_common_pose, count = counts.most_common(1)[0]
        
        # 只有当优势明显时才切换状态，否则保持上一帧（可选）
        return most_common_pose

def main():
    # 加载模型
    model = YOLO("./weights/yolo26s-pose.pt") 
    monitor = HomePoseMonitor()

    cap = cv2.VideoCapture(0)
    
    # 降低分辨率以提高 FPS 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("启动居家姿态监控... 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 使用 track 模式！关键！
        # persist=True 保证 ID 在帧之间保持一致，这是防抖动的前提
        results = model.track(frame, persist=True, stream=True, show=False, verbose=False)

        for r in results:
            annotated_frame = r.plot()
            
            # 检查是否检测到人且 ID 存在
            if r.boxes.id is not None and r.keypoints is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().numpy()
                keypoints = r.keypoints.xy.cpu().numpy()
                confs = r.keypoints.conf.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    kpts = keypoints[i]      # 当前人的关键点
                    box = boxes[i]           # 当前人的框
                    kp_conf = confs[i]       # 关键点置信度

                    # --- 步骤1: 质量过滤 ---
                    if not monitor.is_valid_target(kpts, kp_conf):
                        # 如果目标无效（遮挡严重），绘制灰色框并跳过
                        cv2.putText(annotated_frame, "Occluded/LowConf", 
                                    (int(box[0]), int(box[1])-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)
                        continue

                    # --- 步骤2: 姿态计算 ---
                    raw_pose = monitor.classify_pose(kpts, box)

                    # --- 步骤3: 防抖动平滑 ---
                    final_pose = monitor.smooth_prediction(track_id, raw_pose)

                    # --- 步骤4: 绘制结果 ---
                    # 站立绿色，坐立黄色，躺下红色（警报色）
                    color = (0, 255, 0)
                    if final_pose == "Sitting": color = (0, 255, 255) # Yellow
                    if final_pose == "Lying Down": color = (0, 0, 255) # Red

                    label = f"{final_pose}"
                    cv2.putText(annotated_frame, label, (int(box[0]), int(box[1])-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Home Pose Monitor", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()