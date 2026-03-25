from __future__ import annotations

import math
import os
import json
import warnings
from argparse import ArgumentParser
from importlib import resources
from pathlib import Path
from warnings import filterwarnings

import cv2
import numpy
from PIL import Image
from pyclipper import ET_CLOSEDPOLYGON, JT_ROUND, PyclipperOffset
from shapely.geometry import Polygon
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

filterwarnings("ignore")

__all__ = ["extract_text", "main"]


def _get_weights_dir() -> Path:
    """
    Resolve the packaged weights directory whether running from source or an installed wheel.
    """
    package = __package__ or (__name__.rsplit(".", 1)[0] if "." in __name__ else "vncv")
    try:
        return Path(resources.files(package).joinpath("weights"))
    except Exception:
        return Path(__file__).resolve().parent / "weights"


WEIGHTS_DIR = _get_weights_dir()


def _weight_path(filename: str) -> str:
    path = WEIGHTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Weight file '{filename}' not found in {path.parent}. "
            "Make sure package data is installed correctly."
        )
    return str(path)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sort_polygon(points):
    points.sort(key=lambda x: (x[0][1], x[0][0]))
    for i in range(len(points) - 1):
        for j in range(i, -1, -1):
            if abs(points[j + 1][0][1] - points[j][0][1]) < 10 and \
               (points[j + 1][0][0] < points[j][0][0]):
                temp = points[j]
                points[j] = points[j + 1]
                points[j + 1] = temp
            else:
                break
    return points


def crop_image(image, points):
    assert len(points) == 4, "shape of points must be 4*2"
    crop_width = int(max(numpy.linalg.norm(points[0] - points[1]),
                         numpy.linalg.norm(points[2] - points[3])))
    crop_height = int(max(numpy.linalg.norm(points[0] - points[3]),
                          numpy.linalg.norm(points[1] - points[2])))
    pts_std = numpy.float32([[0, 0],
                             [crop_width, 0],
                             [crop_width, crop_height],
                             [0, crop_height]])
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    image = cv2.warpPerspective(image, matrix, (crop_width, crop_height),
                                borderMode=cv2.BORDER_REPLICATE,
                                flags=cv2.INTER_CUBIC)
    height, width = image.shape[0:2]
    if height * 1.0 / width >= 1.5:
        image = numpy.rot90(image, k=3)
    return image


# ============================================================================
# CTC DECODER
# ============================================================================

class CTCDecoder(object):
    def __init__(self):
        self.character = [
            'blank', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', ';', '<', '=', '>', '?', '@',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z', '[', '\\', ']', '^', '_', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z', '{', '|', '}', '~', '!', '"', '#', '$', '%', '&',
            "'", '(', ')', '*', '+', ',', '-', '.', '/', ' ', ' '
        ]

    def __call__(self, outputs):
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[-1]
        indices = outputs.argmax(axis=2)
        return self.decode(indices, outputs)

    def decode(self, indices, outputs):
        results = []
        confidences = []
        ignored_tokens = [0]
        for i in range(len(indices)):
            selection = numpy.ones(len(indices[i]), dtype=bool)
            selection[1:] = indices[i][1:] != indices[i][:-1]
            for ignored_token in ignored_tokens:
                selection &= indices[i] != ignored_token
            result = []
            confidence = []
            for j in range(len(indices[i][selection])):
                result.append(self.character[indices[i][selection][j]])
                confidence.append(outputs[i][selection][j][indices[i][selection][j]])
            results.append(''.join(result))
            confidences.append(confidence)
        return results, confidences


# ============================================================================
# DETECTION
# ============================================================================

class Detection:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider',
                                                       'CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.min_size = 3
        self.max_size = 960
        self.box_thresh = 0.8
        self.mask_thresh = 0.8
        self.mean = numpy.array([123.675, 116.28, 103.53]).reshape(1, -1).astype('float64')
        self.std = 1 / numpy.array([58.395, 57.12, 57.375]).reshape(1, -1).astype('float64')

    def filter_polygon(self, points, shape):
        width, height = shape[1], shape[0]
        filtered = []
        for point in points:
            if type(point) is list:
                point = numpy.array(point)
            point = self.clockwise_order(point)
            point = self.clip(point, height, width)
            w = int(numpy.linalg.norm(point[0] - point[1]))
            h = int(numpy.linalg.norm(point[0] - point[3]))
            if w <= 3 or h <= 3:
                continue
            filtered.append(point)
        return numpy.array(filtered)

    def boxes_from_bitmap(self, output, mask, dest_width, dest_height):
        mask = (mask * 255).astype(numpy.uint8)
        height, width = mask.shape
        outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[0] if len(outs) == 2 else outs[1]
        boxes, scores = [], []
        for contour in contours:
            points, min_side = self.get_min_boxes(contour)
            if min_side < self.min_size:
                continue
            points = numpy.array(points)
            score = self.box_score(output, contour)
            if self.box_thresh > score:
                continue
            polygon = Polygon(points)
            distance = polygon.area / polygon.length
            offset = PyclipperOffset()
            offset.AddPath(points, JT_ROUND, ET_CLOSEDPOLYGON)
            points = numpy.array(offset.Execute(distance * 1.5)).reshape((-1, 1, 2))
            box, min_side = self.get_min_boxes(points)
            if min_side < self.min_size + 2:
                continue
            box = numpy.array(box)
            box[:, 0] = numpy.clip(numpy.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = numpy.clip(numpy.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return numpy.array(boxes, dtype="int32"), scores

    @staticmethod
    def get_min_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score(bitmap, contour):
        h, w = bitmap.shape[:2]
        contour = contour.copy().reshape(-1, 2)
        x1 = numpy.clip(numpy.min(contour[:, 0]), 0, w - 1)
        y1 = numpy.clip(numpy.min(contour[:, 1]), 0, h - 1)
        x2 = numpy.clip(numpy.max(contour[:, 0]), 0, w - 1)
        y2 = numpy.clip(numpy.max(contour[:, 1]), 0, h - 1)
        mask = numpy.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=numpy.uint8)
        contour[:, 0] -= x1
        contour[:, 1] -= y1
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), color=(1, 1))
        return cv2.mean(bitmap[y1:y2 + 1, x1:x2 + 1], mask)[0]

    @staticmethod
    def clockwise_order(point):
        poly = numpy.zeros((4, 2), dtype="float32")
        s = point.sum(axis=1)
        poly[0] = point[numpy.argmin(s)]
        poly[2] = point[numpy.argmax(s)]
        tmp = numpy.delete(point, (numpy.argmin(s), numpy.argmax(s)), axis=0)
        diff = numpy.diff(numpy.array(tmp), axis=1)
        poly[1] = tmp[numpy.argmin(diff)]
        poly[3] = tmp[numpy.argmax(diff)]
        return poly

    @staticmethod
    def clip(points, h, w):
        for i in range(points.shape[0]):
            points[i, 0] = int(min(max(points[i, 0], 0), w - 1))
            points[i, 1] = int(min(max(points[i, 1], 0), h - 1))
        return points

    def resize(self, image):
        h, w = image.shape[:2]
        ratio = float(self.max_size) / max(h, w) if max(h, w) > self.max_size else 1.0
        resize_h = max(int(round(int(h * ratio) / 32) * 32), 32)
        resize_w = max(int(round(int(w * ratio) / 32) * 32), 32)
        return cv2.resize(image, (resize_w, resize_h))

    @staticmethod
    def zero_pad(image):
        h, w, c = image.shape
        pad = numpy.zeros((max(32, h), max(32, w), c), numpy.uint8)
        pad[:h, :w, :] = image
        return pad

    def __call__(self, x):
        h, w = x.shape[:2]
        if sum([h, w]) < 64:
            x = self.zero_pad(x)
        x = self.resize(x).astype('float32')
        cv2.subtract(x, self.mean, x)
        cv2.multiply(x, self.std, x)
        x = numpy.expand_dims(x.transpose((2, 0, 1)), axis=0)
        outputs = self.session.run(None, {self.inputs.name: x})[0][0, 0]
        boxes, scores = self.boxes_from_bitmap(outputs, outputs > self.mask_thresh, w, h)
        return self.filter_polygon(boxes, (h, w))


# ============================================================================
# CLASSIFICATION
# ============================================================================

class Classification:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider',
                                                       'CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.threshold = 0.98
        self.labels = ['0', '180']

    @staticmethod
    def resize(image):
        input_c, input_h, input_w = 3, 48, 192
        h, w = image.shape[:2]
        ratio = w / float(h)
        resized_w = input_w if math.ceil(input_h * ratio) > input_w else int(math.ceil(input_h * ratio))
        resized_image = cv2.resize(image, (resized_w, input_h)).transpose((2, 0, 1)).astype('float32')
        resized_image = resized_image / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        padded = numpy.zeros((input_c, input_h, input_w), dtype=numpy.float32)
        padded[:, :, 0:resized_w] = resized_image
        return padded

    def __call__(self, images):
        num_images = len(images)
        results = [['', 0.0]] * num_images
        indices = numpy.argsort(numpy.array([x.shape[1] / x.shape[0] for x in images]))
        batch_size = 6
        for i in range(0, num_images, batch_size):
            norm_images = []
            for j in range(i, min(num_images, i + batch_size)):
                norm_images.append(self.resize(images[indices[j]])[numpy.newaxis, :])
            norm_images = numpy.concatenate(norm_images)
            outputs = self.session.run(None, {self.inputs.name: norm_images})[0]
            outputs = [(self.labels[idx], outputs[k, idx])
                       for k, idx in enumerate(outputs.argmax(axis=1))]
            for j, (label, score) in enumerate(outputs):
                results[indices[i + j]] = [label, score]
                if '180' in label and score > self.threshold:
                    images[indices[i + j]] = cv2.rotate(images[indices[i + j]], 1)
        return images, results


# ============================================================================
# ENGLISH RECOGNITION (ONNX)
# ============================================================================

class EnglishRecognition:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider',
                                                       'CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.input_shape = [3, 48, 320]
        self.ctc_decoder = CTCDecoder()

    def resize(self, image, max_wh_ratio):
        input_h, input_w = self.input_shape[1], self.input_shape[2]

        assert self.input_shape[0] == image.shape[2]
        input_w = int((input_h * max_wh_ratio))
        w = self.inputs.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            input_w = w
        h, w = image.shape[:2]
        ratio = w / float(h)
        if math.ceil(input_h * ratio) > input_w:
            resized_w = input_w
        else:
            resized_w = int(math.ceil(input_h * ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padded_image = numpy.zeros((self.input_shape[0], input_h, input_w), dtype=numpy.float32)
        padded_image[:, :, 0:resized_w] = resized_image
        return padded_image

    def __call__(self, images):
        batch_size = 6
        num_images = len(images)

        results = [['', 0.0]] * num_images
        confidences = [['', 0.0]] * num_images
        indices = numpy.argsort(numpy.array([x.shape[1] / x.shape[0] for x in images]))

        for index in range(0, num_images, batch_size):
            input_h, input_w = self.input_shape[1], self.input_shape[2]
            max_wh_ratio = input_w / input_h
            norm_images = []
            for i in range(index, min(num_images, index + batch_size)):
                h, w = images[indices[i]].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)
            for i in range(index, min(num_images, index + batch_size)):
                norm_image = self.resize(images[indices[i]], max_wh_ratio)
                norm_image = norm_image[numpy.newaxis, :]
                norm_images.append(norm_image)
            norm_images = numpy.concatenate(norm_images)

            outputs = self.session.run(None,
                                       {self.inputs.name: norm_images})
            result, confidence = self.ctc_decoder(outputs[0])
            for i in range(len(result)):
                results[indices[index + i]] = result[i]
                confidences[indices[index + i]] = confidence[i]
        return results, confidences


# ============================================================================
# VIETOCR RECOGNITION — thay thế Recognition ONNX cũ
# ============================================================================

class VietOCRRecognition:
    def __init__(self, model_name='vgg_transformer', device='cpu', weight_path=None):
        config = Cfg.load_config_from_name(model_name)
        config['device'] = device
        config['cnn']['pretrained'] = False
        if weight_path and os.path.exists(weight_path):
            config['weights'] = weight_path
        self.predictor = Predictor(config)

    def __call__(self, images):
        results, confidences = [], []
        for img in images:
            # crop_image trả về numpy BGR → convert sang PIL RGB cho VietOCR
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text, conf = self.predictor.predict(pil_img, return_prob=True)
            results.append(text)
            confidences.append(conf)
        return results, confidences


# ============================================================================
# KHỞI TẠO
# ============================================================================

detection = Detection(_weight_path('detection.onnx'))
classification = Classification(_weight_path('classification.onnx'))

_recognition_models = {}

def get_recognition(lang):
    if lang not in _recognition_models:
        if lang == 'vi':
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"[VNCV System] Auto-selected device for recognition: {device.upper()}")
            
            _recognition_models['vi'] = VietOCRRecognition(
                model_name='vgg_transformer',
                device=device,
                weight_path=None     # None = tự download lần đầu
            )

        elif lang == 'en':
            _recognition_models['en'] = EnglishRecognition(_weight_path('recognition.onnx'))
        else:
            raise ValueError(f"Unsupported language: {lang}")
    return _recognition_models[lang]


# ============================================================================
# API & CLI
# ============================================================================

def extract_text(filepath: str,
                 save_annotated: bool = False,
                 annotated_path: str | os.PathLike | None = None,
                 ner: bool = False,
                 lang: str = 'vi',
                 return_dict: bool = False):
    """
    Run OCR on an image and return detected text lines.

    Parameters
    ----------
    filepath: str
        Path to the input image.
    save_annotated: bool
        If True, saves an annotated image with bounding boxes and text labels.
    annotated_path: str | os.PathLike | None
        Optional custom path for the annotated image. Defaults to `<stem>_annotated.png`
        in the same directory as the input file.
    ner: bool
        If True, attempts to export an NER dataset (requires optional helper module).
    """
    image_path = Path(filepath)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image at {filepath}")

    annotated = frame.copy()
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)

    points = detection(frame)
    points = sort_polygon(list(points))

    for point in points:
        cv2.polylines(annotated, [numpy.array(point, dtype=numpy.int32)], True, (0, 255, 0), 2)

    cropped_images = [crop_image(frame, x) for x in points]

    if cropped_images:
        cropped_images, _ = classification(cropped_images)
        rec_model = get_recognition(lang)
        results, confidences = rec_model(cropped_images)
    else:
        results, confidences = [], []

    for i, result in enumerate(results):
        x, y, w, h = cv2.boundingRect(points[i])
        cv2.putText(annotated, result, (int(x), int(y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1, cv2.LINE_AA)

    if save_annotated:
        output_path = Path(annotated_path) if annotated_path else image_path.with_name(
            f"{image_path.stem}_annotated.png")
        cv2.imwrite(str(output_path), annotated)

    if ner:
        try:
            from generate_ner_dataset import process_single_document, save_jsonl
        except ImportError:
            warnings.warn(
                "NER export requested but generate_ner_dataset module is missing.",
                RuntimeWarning
            )
        else:
            doc_id = image_path.stem
            doc = process_single_document(results, doc_id=doc_id)
            if doc:
                save_jsonl([doc], 'dataset.jsonl')
                print(f"[NER] Saved → dataset.jsonl")

    if return_dict:
        output_data = []
        for i, result in enumerate(results):
            box = points[i].tolist()
            conf = confidences[i] if i < len(confidences) else 0.0
            if isinstance(conf, (list, tuple, numpy.ndarray)):
                conf = [float(c) for c in conf]
                conf = sum(conf) / len(conf) if conf else 0.0
            else:
                conf = float(conf)
            output_data.append({
                "text": result,
                "confidence": conf,
                "box": box
            })
        return output_data

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument('filepath', type=str, help='image file path')
    parser.add_argument('--ner', action='store_true',
                        help='export NER dataset sau khi OCR')
    parser.add_argument('--save-annotated', action='store_true',
                        help='save annotated image with bounding boxes')
    parser.add_argument('--annotated-path', type=str, default=None,
                        help='custom output path for annotated image')
    parser.add_argument('--lang', type=str, default='vi', choices=['vi', 'en'],
                        help='language for OCR (vi or en)')
    parser.add_argument('--json', action='store_true',
                        help='output result as json instead of Python list')
    args = parser.parse_args()

    results = extract_text(
        args.filepath,
        save_annotated=args.save_annotated,
        annotated_path=args.annotated_path,
        ner=args.ner,
        lang=args.lang,
        return_dict=args.json
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(results)


if __name__ == '__main__':
    main()
