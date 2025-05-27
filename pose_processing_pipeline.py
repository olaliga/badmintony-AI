import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class FrameProcessor(ABC):
    """
    抽象類別：每一幀的處理器，可 plug-in 多種處理方式
    """
    @abstractmethod
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        pass

class PoseDrawer(FrameProcessor):
    """
    在 frame 上畫出骨架
    """
    def __init__(self, mp_draw, pose_landmarks, pose_connections):
        self.mp_draw = mp_draw
        self.pose_landmarks = pose_landmarks
        self.pose_connections = pose_connections

    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        annotated = frame.copy()
        if context.get('pose_result') and context['pose_result'].pose_landmarks:
            self.mp_draw.draw_landmarks(
                annotated,
                context['pose_result'].pose_landmarks,
                self.pose_connections
            )
        return annotated

class AnglePlotter(FrameProcessor):
    """
    疊加角度折線圖於 frame 右側
    """
    def __init__(self, plot_width, plot_height, angle_names):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.angle_names = angle_names

    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        import matplotlib.pyplot as plt
        angles_list = context.get('angles_list', [])
        frame_idx = context.get('frame_idx', 0)
        total_frames = context.get('total_frames', 1)
        # 畫圖
        plt.figure(figsize=(self.plot_width/100, self.plot_height/100), dpi=100)
        for angle_name in self.angle_names:
            if len(angles_list) > 0 and angle_name in angles_list[0]:
                plt.plot(range(frame_idx+1), [a[angle_name] for a in angles_list], label=angle_name)
        plt.xlim(0, total_frames)
        if len(angles_list) > 0:
            y_min = min([min([a[angle_name] for a in angles_list]) for angle_name in self.angle_names if angle_name in angles_list[0]])
            y_max = max([max([a[angle_name] for a in angles_list]) for angle_name in self.angle_names if angle_name in angles_list[0]])
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
        plot_img = cv2.imread('temp_plot.png', cv2.IMREAD_UNCHANGED)
        if plot_img is None:
            raise ValueError("plot_img 讀取失敗，請檢查 temp_plot.png 是否存在且可讀取")
        if plot_img.shape[2] == 4:
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGRA2BGR)
        plot_img = cv2.resize(plot_img, (self.plot_width, self.plot_height))
        plot_img = plot_img.astype(frame.dtype)
        # 合併
        frame_resized = cv2.resize(frame, (self.plot_height, self.plot_height))
        combined = cv2.hconcat([frame_resized, plot_img])
        return combined

class ScoreOverlay(FrameProcessor):
    """
    疊加分數或其他資訊於 frame
    """
    def __init__(self, text_func):
        self.text_func = text_func  # 傳入一個函式，根據 context 產生要顯示的文字

    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        text = self.text_func(context)
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

class PipelineNode(ABC):
    """
    Pipeline/Workflow 的一個步驟
    """
    @abstractmethod
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class FrameProcessingNode(PipelineNode):
    """
    處理單一幀的 node，可串接多個 FrameProcessor
    """
    def __init__(self, processors: List[FrameProcessor]):
        self.processors = processors

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frame = data['frame']
        for processor in self.processors:
            frame = processor.process(frame, data)
        data['frame'] = frame
        return data

class SaveFrameNode(PipelineNode):
    """
    將 frame 寫入影片
    """
    def __init__(self, writer_key='writer'):
        self.writer_key = writer_key

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        writer = data[self.writer_key]
        frame = data['frame']
        writer.write(frame)
        return data

class Pipeline:
    """
    Pipeline/Workflow 主體，可串接多個 node
    """
    def __init__(self, nodes: List[PipelineNode]):
        self.nodes = nodes

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for node in self.nodes:
            data = node.run(data)
        return data

# --- 使用範例 ---
# 你可以在主程式中這樣組合：
#
# processors = [
#     PoseDrawer(mp_draw, pose_result, mp_pose.POSE_CONNECTIONS),
#     ScoreOverlay(lambda ctx: f"Frame: {ctx['frame_idx']}")
# ]
# pipeline = Pipeline([
#     FrameProcessingNode(processors),
#     SaveFrameNode('writer')
# ])
# for frame in video:
#     data = { ... }
#     pipeline.run(data) 