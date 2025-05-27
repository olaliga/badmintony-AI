import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os
import subprocess
import json
import matplotlib.pyplot as plt

class Badminton2DPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def calculate_angle(self, pointA, pointB, pointC):
        """計算三個點之間的角度"""
        vectorA = np.array([pointA[0] - pointB[0], pointA[1] - pointB[1]])
        vectorB = np.array([pointC[0] - pointB[0], pointC[1] - pointB[1]])
        
        dot = np.inner(vectorA, vectorB)
        mags = np.linalg.norm(vectorA) * np.linalg.norm(vectorB)
        cos = dot / mags
        
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        deg = np.rad2deg(rad)
        
        return int(deg)

    def calculate_vertical_angle(self, pointA, pointB):
        """計算與垂直線的角度"""
        vertical = np.array([0, -1])  # 垂直向下的向量
        vector = np.array([pointB[0] - pointA[0], pointB[1] - pointA[1]])
        
        dot = np.inner(vertical, vector)
        mags = np.linalg.norm(vertical) * np.linalg.norm(vector)
        cos = dot / mags
        
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        deg = np.rad2deg(rad)
        
        return int(deg)

    def extract_landmarks(self, landmarks):
        """提取關鍵點座標"""
        return {
            'left_ankle': [landmarks[27].x, landmarks[27].y, landmarks[27].z],
            'left_knee': [landmarks[25].x, landmarks[25].y, landmarks[25].z],
            'left_hip': [landmarks[23].x, landmarks[23].y, landmarks[23].z],
            'left_shoulder': [landmarks[11].x, landmarks[11].y, landmarks[11].z],
            'left_elbow': [landmarks[13].x, landmarks[13].y, landmarks[13].z],
            'left_wrist': [landmarks[15].x, landmarks[15].y, landmarks[15].z],
            'right_ankle': [landmarks[28].x, landmarks[28].y, landmarks[28].z],
            'right_knee': [landmarks[26].x, landmarks[26].y, landmarks[26].z],
            'right_hip': [landmarks[24].x, landmarks[24].y, landmarks[24].z],
            'right_shoulder': [landmarks[12].x, landmarks[12].y, landmarks[12].z],
            'right_elbow': [landmarks[14].x, landmarks[14].y, landmarks[14].z],
            'right_wrist': [landmarks[16].x, landmarks[16].y, landmarks[16].z]
        }

    def calculate_drive_measurements(self, landmarks):
        """計算平球姿勢的測量值"""
        points = self.extract_landmarks(landmarks)
        
        # 計算膝蓋彎曲角度
        left_knee_angle = self.calculate_angle(
            points['left_hip'],
            points['left_knee'],
            points['left_ankle']
        )
        right_knee_angle = self.calculate_angle(
            points['right_hip'],
            points['right_knee'],
            points['right_ankle']
        )
        
        # 計算上半身軸與法向量的角度
        upper_body_angle = self.calculate_vertical_angle(
            points['right_hip'],
            points['right_shoulder']
        )
        
        # 計算腋下角度
        left_armpit_angle = self.calculate_angle(
            points['left_shoulder'],
            points['left_elbow'],
            points['left_hip']
        )
        right_armpit_angle = self.calculate_angle(
            points['right_shoulder'],
            points['right_elbow'],
            points['right_hip']
        )
        
        # 計算手肘角度
        left_elbow_angle = self.calculate_angle(
            points['left_shoulder'],
            points['left_elbow'],
            points['left_wrist']
        )
        right_elbow_angle = self.calculate_angle(
            points['right_shoulder'],
            points['right_elbow'],
            points['right_wrist']
        )
        
        # 計算手肘位置（相對於肩膀的相對位置）
        left_elbow_position = {
            'x': points['left_elbow'][0] - points['left_shoulder'][0],
            'y': points['left_elbow'][1] - points['left_shoulder'][1]
        }
        right_elbow_position = {
            'x': points['right_elbow'][0] - points['right_shoulder'][0],
            'y': points['right_elbow'][1] - points['right_shoulder'][1]
        }
        
        # 計算上半身軸的位置（相對於髖部的相對位置）
        upper_body_position = {
            'x': points['right_shoulder'][0] - points['right_hip'][0],
            'y': points['right_shoulder'][1] - points['right_hip'][1]
        }
        
        return {
            'angles': {
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'upper_body_angle': upper_body_angle,
                'left_armpit_angle': left_armpit_angle,
                'right_armpit_angle': right_armpit_angle,
                'left_elbow_angle': left_elbow_angle,
                'right_elbow_angle': right_elbow_angle
            },
            'positions': {
                'left_elbow': left_elbow_position,
                'right_elbow': right_elbow_position,
                'upper_body': upper_body_position
            }
        }

    def get_video_rotation(self, video_path):
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate', '-of', 'json', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(result.stdout)
        try:
            rotation = int(info['streams'][0]['tags']['rotate'])
        except (KeyError, IndexError):
            rotation = 90
        return rotation

    def _prepare_image_for_concat(self, img, width, height, dtype):
        """將圖片 resize 成指定寬高、轉成 BGR 3 channel、型態一致"""
        if img is None:
            raise ValueError("圖片讀取失敗，請檢查來源檔案")
        # 若有 alpha channel，轉成 BGR
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (width, height))
        img = img.astype(dtype)
        return img

    def analyze_drive_video(self, video_path, with_plot=True, angle_names=None):
        """
        分析平球姿勢影片，並可選擇同步輸出折線圖動畫合成影片。
        """
        import matplotlib.pyplot as plt
        if angle_names is None:
            angle_names = ['left_knee_angle', 'right_knee_angle', 'left_elbow_angle']

        if not os.path.exists(video_path):
            raise ValueError(f"找不到影片檔案：{video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法讀取影片：{video_path}")

        # 影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 輸出資料夾
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, extension = os.path.splitext(os.path.basename(video_path))
        result_dir = os.path.join('results', f'pose_analysis_{filename}')
        os.makedirs(result_dir, exist_ok=True)

        # 輸出影片
        output_video_path = os.path.join(result_dir, 'annotated_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        rotation = self.get_video_rotation(video_path)
        if rotation in [90, 270]:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (height, width))
            plot_width, plot_height = height, width
        else:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            plot_width, plot_height = width, height

        # 合成影片（骨架+折線圖）
        if with_plot:
            output_combined_path = os.path.join(result_dir, 'combined_video.mp4')
            out_combined = cv2.VideoWriter(output_combined_path, fourcc, fps, (2*plot_width, plot_height))

        # 儲存所有幀的測量數據
        angles_list = []
        positions_list = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 根據 rotation 旋轉
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)

            if results.pose_landmarks:
                measurements = self.calculate_drive_measurements(results.pose_landmarks.landmark)
                angles_list.append(measurements['angles'])
                positions_list.append(measurements['positions'])
                # 標記骨架
                annotated_frame = frame.copy()
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
            else:
                annotated_frame = frame.copy()

            # 寫入骨架影片
            out.write(annotated_frame)

            # --- 合成影片（骨架+折線圖）---
            if with_plot:
                # 動態繪製折線圖
                plt.figure(figsize=(plot_width/100, plot_height/100), dpi=100)
                for angle_name in angle_names:
                    if len(angles_list) > 0 and angle_name in angles_list[0]:
                        plt.plot(range(frame_idx+1), [a[angle_name] for a in angles_list], label=angle_name)
                plt.xlim(0, total_frames)
                if len(angles_list) > 0:
                    y_min = min([min([a[angle_name] for a in angles_list]) for angle_name in angle_names if angle_name in angles_list[0]])
                    y_max = max([max([a[angle_name] for a in angles_list]) for angle_name in angle_names if angle_name in angles_list[0]])
                else:
                    y_min, y_max = 0, 1
                plt.ylim(y_min-10, y_max+10)
                plt.xlabel('Frame')
                plt.ylabel('Angle (deg)')
                plt.title('Movement Angle Variation')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('temp_plot.png')
                plt.close()
                # 準備兩張圖
                plot_img = cv2.imread('temp_plot.png', cv2.IMREAD_UNCHANGED)
                annotated_frame_for_concat = self._prepare_image_for_concat(annotated_frame, plot_width, plot_height, annotated_frame.dtype)
                plot_img_for_concat = self._prepare_image_for_concat(plot_img, plot_width, plot_height, annotated_frame.dtype)
                combined = cv2.hconcat([annotated_frame_for_concat, plot_img_for_concat])
                out_combined.write(combined)

            frame_idx += 1

        cap.release()
        out.release()
        if with_plot:
            out_combined.release()
            if os.path.exists('temp_plot.png'):
                os.remove('temp_plot.png')

        # 儲存測量數據到CSV
        df_angles = pd.DataFrame(angles_list)
        df_positions = pd.DataFrame(positions_list)
        csv_angles_path = os.path.join(result_dir, 'drive_angles.csv')
        csv_positions_path = os.path.join(result_dir, 'drive_positions.csv')
        df_angles.to_csv(csv_angles_path, index=False)
        df_positions.to_csv(csv_positions_path, index=False)

        return {
            'annotated_video_path': output_video_path,
            'csv_angles_path': csv_angles_path,
            'csv_positions_path': csv_positions_path,
            'total_frames': frame_idx,
            'combined_video_path': output_combined_path if with_plot else None
        }

def main():
    # 建立results資料夾（如果不存在）
    os.makedirs('results', exist_ok=True)
    
    # 初始化分析器
    analyzer = Badminton2DPoseAnalyzer()
    
    # 請使用者輸入影片路徑
    # video_path = input("請輸入平球姿勢影片路徑：")
    video_paths = [os.path.join('TestVideo', file) for file in os.listdir('TestVideo') if file.endswith('.mp4') or file.endswith('.MOV')  ]   

    try:
        for video_path in video_paths[:3]:
            print(f"分析影片：{video_path}")
            # 分析影片
            results = analyzer.analyze_drive_video(video_path)
            print("\n分析完成！")
            print(f"標記後的影片已儲存至：{results['annotated_video_path']}")
            print(f"角度數據已儲存至：{results['csv_angles_path']}")
            print(f"位置數據已儲存至：{results['csv_positions_path']}")
            
            # # 顯示角度數據
            # print("\n角度測量：")
            # for angle_name, angle_value in results['measurements']['angles'].items():
            #     print(f"{angle_name}: {angle_value}°")
                
            # # 顯示位置數據
            # print("\n位置測量：")
            # for pos_name, pos_value in results['measurements']['positions'].items():
            #     print(f"{pos_name}: x={pos_value['x']:.2f}, y={pos_value['y']:.2f}")
            
    except Exception as e:
        print(f"錯誤：{str(e)}")

if __name__ == "__main__":
    main() 