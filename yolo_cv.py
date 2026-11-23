"""
YOLOv8 检测器 — 纯 OpenCV DNN 实现（不依赖 ultralytics / torch）
用于 APK 打包（python-for-android 没有 ultralytics recipe）
"""
import cv2
import numpy as np


# ==========================================================
# 🇨🇳 国服中文卡牌名
# ==========================================================
CN_NAMES = {
    0: '弓箭手', 1: '野蛮人', 2: '攻城槌', 3: '炸弹兵',
    4: '格斗手', 5: '加农炮', 6: '巨人', 7: '哥布林',
    8: '哥布林笼', 9: '哥布林小屋', 10: '骑士',
    11: '超级亡灵', 12: '迷你皮卡', 13: '亡灵',
    14: '火枪手', 15: '骷髅', 16: '矛哥布林',
    17: '墓碑', 18: '女武神',
}

# 英文兜底（训练集标签）
EN_NAMES = {
    0: 'Archer', 1: 'Barbarian', 2: 'Battle_Ram', 3: 'Bomber',
    4: 'Brawler', 5: 'Cannon', 6: 'Giant', 7: 'Goblin',
    8: 'Goblin_Cage', 9: 'Goblin_Hut', 10: 'Knight',
    11: 'Mega_Minion', 12: 'Mini_PEKKA', 13: 'Minion',
    14: 'Musketeer', 15: 'Skeleton', 16: 'Spear_Goblin',
    17: 'Tombstone', 18: 'Valkyrie',
}

# 国服阵营配色（便于人眼快速区分）
CLASS_COLORS = [
    (64, 200, 255), (220, 140, 60), (80, 160, 220), (180, 200, 80),
    (255, 160, 70), (120, 200, 220), (90, 90, 230), (110, 230, 120),
    (200, 130, 60), (150, 120, 200), (70, 190, 255), (230, 100, 160),
    (100, 200, 130), (210, 170, 90), (170, 200, 110), (230, 230, 230),
    (120, 200, 90), (160, 160, 210), (220, 120, 140),
]


class YOLOv8Detector:
    """用 cv2.dnn 跑 YOLOv8 ONNX，无 torch 依赖。"""

    def __init__(self, onnx_path='best.onnx', conf_th=0.40, nms_th=0.45,
                 input_size=640, use_cn=True, device='AUTO'):
        if not cv2.data.haarcascades and False:
            pass
        self.input_size = int(input_size)
        self.conf_th = float(conf_th)
        self.nms_th = float(nms_th)
        self.use_cn = use_cn
        self.names = CN_NAMES if use_cn else EN_NAMES
        self.class_names = list(self.names.values())

        # 优先 OpenCV 原生读取；失败再走 onnxruntime 兜底
        self.net = None
        self.backend = None
        try:
            net = cv2.dnn.readNetFromONNX(onnx_path)
            # 选可用后端
            target = {
                'AUTO': cv2.dnn.DNN_TARGET_CPU,
                'CPU': cv2.dnn.DNN_TARGET_CPU,
            }.get(str(device).upper(), cv2.dnn.DNN_TARGET_CPU)
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(target)
            except Exception:
                pass
            self.net = net
            self.backend = 'opencv'
        except Exception as e:
            print(f'[检测器] OpenCV DNN 加载失败，尝试 onnxruntime: {e}')
            self.net = None
            self.backend = None

        if self.net is None:
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.log_severity_level = 3
                providers = ['CPUExecutionProvider']
                self.ort = ort.InferenceSession(onnx_path, sess_options=so,
                                                providers=providers)
                self.input_name = self.ort.get_inputs()[0].name
                self.backend = 'onnxruntime'
                print('[检测器] 使用 onnxruntime CPU')
            except Exception as e:
                raise RuntimeError(
                    f'两种推理后端都不可用: {e}'
                )
        else:
            self.ort = None
            print('[检测器] 使用 OpenCV DNN')

        print(f'[检测器] 就绪 · {len(self.class_names)} 类 · '
              f'输入 {self.input_size} · conf>{self.conf_th}')

    # ------------------------------------------------------
    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        r = min(self.input_size / h, self.input_size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        dw, dh = self.input_size - nw, self.input_size - nh
        left, right = dw // 2, dw - dw // 2
        top, bottom = dh // 2, dh - dh // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, r, (left, top)

    # ------------------------------------------------------
    def _run(self, padded):
        if self.backend == 'opencv':
            blob = cv2.dnn.blobFromImage(
                padded, 1 / 255.0, (self.input_size, self.input_size),
                swapRB=True, crop=False
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            # YOLOv8 输出 (1, 4+nc, N) 或 (1, N, 4+nc)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            out = np.squeeze(outputs)
            if out.shape[0] <= 84 or out.shape[0] < out.shape[-1]:
                out = out.T
            return out
        else:
            blob = padded.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None, ...]
            outputs = self.ort.run(None, {self.input_name: blob})
            out = np.squeeze(outputs[0])
            if out.shape[0] <= 84 or out.shape[0] < out.shape[-1]:
                out = out.T
            return out

    # ------------------------------------------------------
    def _decode(self, out):
        """解析 YOLOv8 输出：每行 [cx,cy,w,h, conf1..confN]"""
        rows = []
        for det in out:
            cls_scores = det[4:]
            if cls_scores.size == 0:
                continue
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < self.conf_th:
                continue
            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            rows.append([x1, y1, x2, y2, conf, cls_id])

        if not rows:
            return []

        rows = np.array(rows, dtype=np.float32)
        boxes = rows[:, :4].tolist()
        scores = rows[:, 4].tolist()
        idxs = cv2.dnn.NMSBoxes(boxes, scores,
                                self.conf_th, self.nms_th)
        if idxs is None or len(idxs) == 0:
            return []

        if isinstance(idxs, np.ndarray):
            idxs = idxs.flatten()
        out_det = []
        for i in idxs:
            x1, y1, x2, y2, conf, cls_id = rows[i]
            out_det.append({
                'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(conf),
                'cls': int(cls_id),
                'name_cn': self.names.get(int(cls_id), f'类别{int(cls_id)}'),
            })
        return out_det

    # ------------------------------------------------------
    def detect(self, frame):
        """返回检测框列表，坐标为原图坐标"""
        if frame is None:
            return []
        padded, r, (left, top) = self._preprocess(frame)
        out = self._run(padded)
        dets = self._decode(out)

        for d in dets:
            x1, y1, x2, y2 = d['xyxy']
            x1 = (x1 - left) / r
            y1 = (y1 - top) / r
            x2 = (x2 - left) / r
            y2 = (y2 - top) / r
            h, w = frame.shape[:2]
            d['xyxy'] = [
                float(max(0, min(x1, w))),
                float(max(0, min(y1, h))),
                float(max(0, min(x2, w))),
                float(max(0, min(y2, h))),
            ]
        return dets

    # ------------------------------------------------------
    def draw(self, frame, dets, show_conf=True):
        """在帧上画中文标签和检测框"""
        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d['xyxy']]
            color = CLASS_COLORS[d['cls'] % len(CLASS_COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            text = d['name_cn']
            if show_conf:
                text += f' {d["conf"]*100:.0f}%'
            fs = 0.65
            th = 2
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            ty = y1 - 6 if y1 - tht - 10 > 0 else y1 + tht + 16
            cv2.rectangle(vis, (x1, ty - tht - 8), (x1 + tw + 8, ty + 4),
                          color, -1)
            cv2.putText(vis, text, (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th,
                        cv2.LINE_AA)
        return vis