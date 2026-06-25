
"""
Система автоматического подсчёта товаров на полке.
Реализована на основе YOLOv8s — лучшей модели по итогам
сравнительного анализа 5 архитектур на датасете SKU-110K.
 
Запуск: python app.py
Или в Google Colab: скопировать код в ячейку и выполнить.
 
Требования:
  - ultralytics
  - gradio
  - Веса модели: best.pt (YOLOv8s, обученная на SKU-110K)
  - Тестовые изображения + YOLO-лейблы (для режима оценки)
"""
 
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import time
import os
import gradio as gr
 
 
# ======================== Вспомогательные функции ========================
 
def compute_iou_matrix(boxes_a, boxes_b):
    """IoU матрица между двумя массивами боксов [N,4] и [M,4] в формате xyxy."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)
 
 
def load_gt_for_image(img_name, img_size=640):
    """Загружает GT боксы из YOLO-txt и пересчитывает в xyxy пиксельные координаты."""
    label_path = os.path.join(TEST_LABELS_DIR, img_name.rsplit('.', 1)[0] + '.txt')
    if not os.path.exists(label_path):
        return np.zeros((0, 4))
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * img_size
            y1 = (cy - bh / 2) * img_size
            x2 = (cx + bw / 2) * img_size
            y2 = (cy + bh / 2) * img_size
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.zeros((0, 4))
 
 
# ======================== Пути (подставьте свои) ========================
 
WEIGHTS_PATH = '/content/drive/MyDrive/shelf_detection/models/yolov8s/weights/best.pt'
TEST_IMAGES_DIR = '/content/yolo_dataset/images/test'
TEST_LABELS_DIR = '/content/yolo_dataset/labels/test'
 
 
# ======================== Класс детектора ========================
 
class ShelfProductDetector:
    """
    Финальный детектор товаров на полке.
    Использует YOLOv8s — лучшую модель по сравнительному анализу.
    """
 
    def __init__(self, weights_path=WEIGHTS_PATH):
        self.model = YOLO(weights_path)
        self.model_name = 'YOLOv8s'
 
    def detect(self, image, conf_thresh=0.25, img_size=640):
        """Детекция товаров на одном изображении."""
        t0 = time.time()
        results = self.model(
            image, conf=conf_thresh, imgsz=img_size, verbose=False, max_det=300
        )
        elapsed_ms = (time.time() - t0) * 1000
 
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), 0, elapsed_ms
 
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        return boxes, scores, len(boxes), elapsed_ms
 
    def draw_detections(self, image, boxes, scores, count):
        """Рисует все детекции зелёным с подписью количества."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline='lime', width=2)
 
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28
            )
        except OSError:
            font = ImageFont.load_default()
 
        text = f'Найдено товаров: {count}'
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle([bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5], fill='black')
        draw.text((10, 10), text, fill='lime', font=font)
        return img
 
    def evaluate_with_gt(self, image, boxes, scores, gt_boxes, iou_thresh=0.5):
        """Сравнивает с эталоном. Рисует TP/FP/FN разными цветами."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
 
        tp_boxes, fp_boxes = [], []
        matched_gt = set()
        if len(boxes) > 0 and len(gt_boxes) > 0:
            iou_mat = compute_iou_matrix(boxes, gt_boxes)
            order = np.argsort(-scores)
            for pi in order:
                best_gt, best_iou = -1, iou_thresh
                for gi in range(len(gt_boxes)):
                    if gi in matched_gt:
                        continue
                    if iou_mat[pi, gi] > best_iou:
                        best_iou = iou_mat[pi, gi]
                        best_gt = gi
                if best_gt >= 0:
                    tp_boxes.append(boxes[pi])
                    matched_gt.add(best_gt)
                else:
                    fp_boxes.append(boxes[pi])
        else:
            fp_boxes = list(boxes)
        fn_boxes = [gt_boxes[gi] for gi in range(len(gt_boxes)) if gi not in matched_gt]
 
        for box in tp_boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline='lime', width=2)
        for box in fp_boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
        for box in fn_boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline='yellow', width=3)
 
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22
            )
        except OSError:
            font = ImageFont.load_default()
 
        accuracy = len(tp_boxes) / max(len(gt_boxes), 1) * 100
        text_lines = [
            f'Эталон: {len(gt_boxes)}  Найдено: {len(boxes)}',
            f'TP={len(tp_boxes)}  FP={len(fp_boxes)}  FN={len(fn_boxes)}',
            f'Точность подсчёта: {accuracy:.1f}%',
        ]
        y_pos = 10
        for line in text_lines:
            bbox = draw.textbbox((10, y_pos), line, font=font)
            draw.rectangle([bbox[0] - 5, bbox[1] - 2, bbox[2] + 5, bbox[3] + 2], fill='black')
            draw.text((10, y_pos), line, fill='white', font=font)
            y_pos += 30
 
        return img, len(tp_boxes), len(fp_boxes), len(fn_boxes), accuracy
 
 
# ======================== Инициализация ========================
 
detector = ShelfProductDetector()
print(f'Детектор загружен: {detector.model_name}')
 
 
# ======================== Gradio-интерфейс ========================
 
test_files_list = sorted(os.listdir(TEST_IMAGES_DIR))
 
 
def mode_inference(image, conf_thresh):
    """Режим 1: детекция + подсчёт на произвольной картинке."""
    if image is None:
        return None, 'Загрузите изображение'
 
    pil_img = image.resize((640, 640), Image.BILINEAR)
    boxes, scores, count, ms = detector.detect(pil_img, conf_thresh=conf_thresh)
    result_img = detector.draw_detections(pil_img, boxes, scores, count)
 
    report = (
        f'**Найдено товаров:** {count}\n\n'
        f'**Время инференса:** {ms:.1f} мс\n\n'
        f'**Модель:** {detector.model_name} '
        f'(наилучшая по нашему сравнительному анализу 5 архитектур)\n\n'
        f'**Порог уверенности:** {conf_thresh}'
    )
    return result_img, report
 
 
def mode_evaluation_by_name(img_name, conf_thresh):
    """Режим 2: оценка тестовой картинки по имени."""
    if not img_name:
        return None, 'Выберите имя файла'
 
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        return None, f'Файл {img_name} не найден'
 
    pil_img = Image.open(img_path).convert('RGB').resize((640, 640), Image.BILINEAR)
    gt = load_gt_for_image(img_name, img_size=640)
 
    boxes, scores, count, ms = detector.detect(pil_img, conf_thresh=conf_thresh)
    result_img, tp, fp, fn, acc = detector.evaluate_with_gt(pil_img, boxes, scores, gt)
 
    report = (
        f'**Файл:** {img_name}\n\n'
        f'**Эталон (ground truth):** {len(gt)} товаров\n\n'
        f'**Модель нашла:** {count} товаров\n\n'
        f'**TP (правильно):** {tp}\n\n'
        f'**FP (ложные срабатывания):** {fp}\n\n'
        f'**FN (пропуски):** {fn}\n\n'
        f'**Точность подсчёта:** {acc:.1f}%\n\n'
        f'**Время инференса:** {ms:.1f} мс'
    )
    return result_img, report
 
 
with gr.Blocks(title='Подсчёт товаров на полке') as demo:
    gr.Markdown(
        '# Система автоматического подсчёта товаров на полке\n'
        'Реализована на основе YOLOv8s — лучшей модели по итогам '
        'сравнительного анализа 5 архитектур.'
    )
 
    with gr.Tab('Режим инференса (произвольное фото)'):
        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(type='pil', label='Загрузить фото полки')
                inp_conf = gr.Slider(
                    0.1, 0.9, value=0.25, step=0.05, label='Порог уверенности'
                )
                btn_run = gr.Button('Найти и посчитать', variant='primary')
            with gr.Column():
                out_img = gr.Image(type='pil', label='Результат')
                out_text = gr.Markdown()
        btn_run.click(
            mode_inference, inputs=[inp_img, inp_conf], outputs=[out_img, out_text]
        )
 
    with gr.Tab('Режим оценки (тестовое фото с эталоном)'):
        gr.Markdown(
            'Выберите тестовое изображение из SKU-110K. '
            'Программа сравнит предсказания модели с эталонной разметкой '
            'и подсветит ошибки.'
        )
        with gr.Row():
            with gr.Column():
                eval_name = gr.Dropdown(
                    choices=test_files_list, label='Тестовое изображение'
                )
                eval_conf = gr.Slider(
                    0.1, 0.9, value=0.25, step=0.05, label='Порог уверенности'
                )
                btn_eval = gr.Button('Оценить', variant='primary')
            with gr.Column():
                out_eval_img = gr.Image(type='pil', label='Результат с TP/FP/FN')
                out_eval_text = gr.Markdown()
        btn_eval.click(
            mode_evaluation_by_name,
            inputs=[eval_name, eval_conf],
            outputs=[out_eval_img, out_eval_text],
        )
 
    gr.Markdown(
        '---\n'
        '🟢 **Зелёный** — TP (правильно)  '
        '🔴 **Красный** — FP (ложное срабатывание)  '
        '🟡 **Жёлтый** — FN (пропуск)'
    )
 
demo.launch(share=True, debug=False)
