"""
国服专用 Clash Royale 实时识别 APK 主程序
Kivy UI + OpenCV DNN(ONNX) 推理，无 torch / ultralytics 依赖
"""
import os
import time
from threading import Thread

import cv2
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

# ==========================================================
# 模型与资源路径（Android 私有目录 / 当前目录）
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _request_android_permissions():
    """Android 6+ 需要在运行时申请相机权限。"""
    try:
        from android.permissions import request_permissions, Permission
        request_permissions([Permission.CAMERA,
                             Permission.WRITE_EXTERNAL_STORAGE,
                             Permission.READ_EXTERNAL_STORAGE])
        print('[权限] 已发起运行时权限申请')
    except Exception as exc:
        print(f'[权限] 跳过（非 Android 或模块缺失）: {exc}')


def _find_asset(name):
    """在多个可能位置查找资源文件。"""
    candidates = [
        os.path.join(BASE_DIR, name),
        os.path.join(os.getcwd(), name),
    ]
    if hasattr(App, 'user_data_dir'):
        try:
            candidates.append(os.path.join(App.user_data_dir, name))
        except Exception:
            pass
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, name)


MODEL_PATH = _find_asset('best.onnx')

# 国服中文卡牌名（与 best.pt / best.onnx 的 19 类一一对应）
CN_CARD_NAMES = {
    0: '弓箭手', 1: '野蛮人', 2: '攻城槌', 3: '炸弹兵',
    4: '格斗手', 5: '加农炮', 6: '巨人', 7: '哥布林',
    8: '哥布林笼', 9: '哥布林小屋', 10: '骑士',
    11: '超级亡灵', 12: '迷你皮卡', 13: '亡灵',
    14: '火枪手', 15: '骷髅', 16: '矛哥布林',
    17: '墓碑', 18: '女武神',
}

# 国服卡组配色，方便人眼区分
CARD_COLORS = [
    (64, 200, 255), (220, 140, 60), (80, 160, 220), (180, 200, 80),
    (255, 160, 70), (120, 200, 220), (90, 90, 230), (110, 230, 120),
    (200, 130, 60), (150, 120, 200), (70, 190, 255), (230, 100, 160),
    (100, 200, 130), (210, 170, 90), (170, 200, 110), (230, 230, 230),
    (120, 200, 90), (160, 160, 210), (220, 120, 140),
]

CONF_TH = 0.30
INPUT_SIZE = 640
TARGET_FPS = 30


# ==========================================================
# 检测器：OpenCV DNN + ONNX
# ==========================================================
class CRDetector:
    """封装 cv2.dnn 推理，中文标签绘制。"""

    def __init__(self, model_path=MODEL_PATH, conf_th=CONF_TH,
                 input_size=INPUT_SIZE):
        self.conf_th = float(conf_th)
        self.input_size = int(input_size)
        self.net = None
        self.backend = None
        self.ready = False

        if not os.path.exists(model_path):
            print(f'[检测器] 模型不存在: {model_path}')
            return

        try:
            net = cv2.dnn.readNetFromONNX(model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.net = net
            self.backend = 'opencv-dnn'
            self.ready = True
            print(f'[检测器] 就绪 · 19 类 · {self.input_size}px · '
                  f'conf>{self.conf_th} · {os.path.basename(model_path)}')
        except Exception as exc:
            print(f'[检测器] 初始化失败: {exc}')
            self.ready = False

    # --------------------------------------------------
    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        ratio = min(self.input_size / h, self.input_size / w)
        nh, nw = int(round(h * ratio)), int(round(w * ratio))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        dw, dh = self.input_size - nw, self.input_size - nh
        left, right = dw // 2, dw - dw // 2
        top, bottom = dh // 2, dh - dh // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT,
                                    value=(114, 114, 114))
        return padded, ratio, left, top

    # --------------------------------------------------
    def detect(self, frame):
        """返回 [{xyxy, conf, cls, name}]，坐标为原图尺度。"""
        if not self.ready or frame is None or frame.size == 0:
            return []

        padded, ratio, pad_l, pad_t = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            padded, 1 / 255.0,
            (self.input_size, self.input_size),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        output = self.net.forward()

        if isinstance(output, (list, tuple)):
            output = output[0]
        out = np.squeeze(output)

        # YOLOv8 输出 (1, 4+nc, N) 或已转置为 (1, N, 4+nc)
        if out.ndim == 2 and out.shape[0] < out.shape[1]:
            out = out.T

        results = []
        for det in out:
            scores = det[4:]
            if scores.size == 0:
                continue
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < self.conf_th:
                continue

            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            results.append({
                'xyxy': [cx - bw / 2, cy - bh / 2,
                         cx + bw / 2, cy + bh / 2],
                'conf': conf,
                'cls': cls_id,
            })

        if not results:
            return []

        boxes = [[float(v) for v in r['xyxy']] for r in results]
        scores = [r['conf'] for r in results]
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_th, 0.45)
        if idxs is None or len(idxs) == 0:
            return []
        if isinstance(idxs, np.ndarray):
            idxs = idxs.flatten().tolist()

        h, w = frame.shape[:2]
        dets = []
        for i in idxs:
            r = results[int(i)]
            x1, y1, x2, y2 = r['xyxy']
            x1 = max(0.0, min(float((x1 - pad_l) / ratio), w))
            y1 = max(0.0, min(float((y1 - pad_t) / ratio), h))
            x2 = max(0.0, min(float((x2 - pad_l) / ratio), w))
            y2 = max(0.0, min(float((y2 - pad_t) / ratio), h))
            dets.append({
                'xyxy': [x1, y1, x2, y2],
                'conf': r['conf'],
                'cls': r['cls'],
                'name': CN_CARD_NAMES.get(r['cls'], f'类别{r["cls"]}'),
            })
        return dets

    # --------------------------------------------------
    def draw(self, frame, dets):
        """画中文标签 + 检测框。"""
        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d['xyxy']]
            color = CARD_COLORS[d['cls'] % len(CARD_COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # cv2.putText 不支持中文，画 ASCII 编号，中文名在界面下方显示
            text = f"#{d['cls']} {d['conf'] * 100:.0f}%"
            fs, th = 0.6, 2
            (tw, tht), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            ty = y1 - 6 if y1 - tht - 10 > 0 else y1 + tht + 16
            cv2.rectangle(vis, (x1, ty - tht - 8), (x1 + tw + 8, ty + 4),
                          color, -1)
            cv2.putText(vis, text, (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th,
                        cv2.LINE_AA)
        return vis


# ==========================================================
# Kivy UI
# ==========================================================
class CRAILayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=6, padding=6,
                         **kwargs)

        self.detector = CRDetector()
        self.running = False
        self.worker = None
        self.last_vis = None

        # 状态条
        self.status_label = Label(
            text='模型未就绪' if not self.detector.ready else '待机',
            size_hint_y=None, height=28,
            font_size='13sp', color=(1, 1, 1, 1))
        self.add_widget(self.status_label)

        # 预览
        self.image_view = Image(size_hint=(1, 0.72))
        self.add_widget(self.image_view)
        self._paint_placeholder()

        # 控制按钮
        btn_row = BoxLayout(size_hint=(1, None), height=52, spacing=8)
        self.start_btn = Button(text='开始识别')
        self.start_btn.bind(on_press=self.start_detection)
        self.stop_btn = Button(text='停止')
        self.stop_btn.bind(on_press=self.stop_detection)
        self.stop_btn.disabled = True
        btn_row.add_widget(self.start_btn)
        btn_row.add_widget(self.stop_btn)
        self.add_widget(btn_row)

        # 统计
        self.stats_label = Label(
            text='', size_hint_y=None, height=76,
            font_size='10sp', halign='left', valign='top')
        self.add_widget(self.stats_label)

        # 初始化模型（异步，避免阻塞首帧）
        if not self.detector.ready:
            Clock.schedule_once(self._retry_init, 1.0)

    # --------------------------------------------------
    def _retry_init(self, dt):
        self.detector = CRDetector()
        self.status_label.text = ('模型就绪' if self.detector.ready
                                  else '模型缺失: best.onnx')

    # --------------------------------------------------
    def _paint_placeholder(self):
        """无画面时显示占位图。"""
        w, h = 640, 400
        canvas = np.full((h, w, 3), 26, dtype=np.uint8)
        cv2.putText(canvas, 'Clash Royale', (w // 2 - 180, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 200, 90), 3,
                    cv2.LINE_AA)
        cv2.putText(canvas, 'guofu detect', (w // 2 - 150, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 190, 255), 2,
                    cv2.LINE_AA)
        self._show_frame(canvas)

    # --------------------------------------------------
    def _show_frame(self, bgr):
        if bgr is None or bgr.size == 0:
            return
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        tex = Texture.create(size=(rgba.shape[1], rgba.shape[0]),
                             colorfmt='rgba')
        tex.flip_horizontal()
        tex.blit_buffer(rgba.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.image_view.texture = tex

    # --------------------------------------------------
    def start_detection(self, _instance=None):
        if not self.detector.ready:
            self.status_label.text = '模型未就绪，无法开始'
            return
        if self.running:
            return
        self.running = True
        self.status_label.text = '识别中'
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.worker = Thread(target=self._detection_loop, daemon=True)
        self.worker.start()

    # --------------------------------------------------
    def stop_detection(self, _instance=None):
        self.running = False
        self.status_label.text = '已停止'
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.stats_label.text = ''

    # --------------------------------------------------
    def _open_camera(self):
        """Android 优先用后排摄像头；失败则回退 0。"""
        for idx in (0, 1, 2):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    # 少缓冲，降低延迟
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return cap, idx
                cap.release()
            except Exception as exc:
                print(f'[摄像头] index={idx} 打开异常: {exc}')
        return None, None

    # --------------------------------------------------
    def _detection_loop(self):
        try:
            cap, cam_idx = self._open_camera()
        except Exception as exc:
            print(f'[摄像头] 初始化失败: {exc}')
            cap, cam_idx = None, None
        if cap is None:
            Clock.schedule_once(
                lambda dt: setattr(self.status_label, 'text',
                                   '摄像头不可用'), 0)
            self.running = False
            Clock.schedule_once(
                lambda dt: setattr(self.stop_btn, 'disabled', True), 0)
            Clock.schedule_once(
                lambda dt: setattr(self.start_btn, 'disabled', False), 0)
            return

        print(f'[摄像头] 已打开 index={cam_idx}')
        interval = 1.0 / float(TARGET_FPS)
        frames, t0, fps = 0, time.time(), 0.0
        infer_ms = 0.0

        while self.running:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            t_start = time.time()
            dets = self.detector.detect(frame)
            infer_ms = (time.time() - t_start) * 1000.0
            vis = self.detector.draw(frame, dets)

            frames += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = frames / (now - t0)
                frames, t0 = 0, now

            self._push_to_ui(vis, dets, fps, infer_ms)

            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        cap.release()
        print('[摄像头] 已释放')

    # --------------------------------------------------
    def _push_to_ui(self, vis, dets, fps, infer_ms):
        """线程安全地把结果交给 UI 线程。"""
        Clock.schedule_once(
            lambda dt: self._apply_ui(vis, dets, fps, infer_ms), 0)

    # --------------------------------------------------
    def _apply_ui(self, vis, dets, fps, infer_ms):
        self._show_frame(vis)
        names = sorted({d['name'] for d in dets})
        hot = '  '.join(names) if names else '无目标'
        legend = ' '.join(f"#{k}={v}" for k, v in sorted(CN_CARD_NAMES.items()))
        self.stats_label.text = (
            f'FPS {fps:4.1f}  推理 {infer_ms:4.0f}ms  目标 {len(dets):2d}\n'
            f'{hot}\n{legend}')

    # --------------------------------------------------
    def on_stop(self):
        self.running = False


class CRAIApp(App):
    title = '国服CR实时识别'

    def build(self):
        _request_android_permissions()
        self.layout = CRAILayout()
        return self.layout

    def on_pause(self):
        if hasattr(self, 'layout'):
            self.layout.stop_detection()
        return True

    def on_resume(self):
        return True

    def on_stop(self):
        if hasattr(self, 'layout'):
            self.layout.on_stop()


if __name__ == '__main__':
    CRAIApp().run()