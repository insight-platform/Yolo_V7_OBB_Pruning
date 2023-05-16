from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from cvd.datasets.annotations.objects import PredictedObject
from cvd.datasets.annotations.rbbox import RBBoxXYCenterWHA
from cvd.datasets.image_dataset import ImagesDataset
from cvd.models.interface import PredictionModel
from cvd.datasets.annotations.image_annotation import ImageAnnotation
from cvd.datasets.image_dataset_item import ImageDatasetItem

from yolov7obb.utils.datasets import letterbox
from yolov7obb.models.experimental import attempt_load
from yolov7obb.utils.general import non_max_suppression_obb
from yolov7obb.utils.torch_utils import select_device
import cv2
from tqdm.auto import tqdm

class YOLOV7OBB(PredictionModel):

    def __init__(
            self,
            input_image_size: Tuple[int, int] = (640, 640),
            det_conf_threshold: float = 0.2,
            iou_threshold: float = 0.4,
            labels: Optional[Dict[int, str]] = None,
            single_class: bool = False
    ):
        """

        :param input_image_size: Input image size for inference (width, height)
        :param det_conf_threshold: Detections with a confident less than this will be rejected.
        :param iou_threshold: Intersection over union threshold for NMS
        :param labels: Mapping numerical value of class into string label
        """
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = None
        self.det_conf_threshold = det_conf_threshold
        self.input_image_size = input_image_size
        self.iou_threshold = iou_threshold
        self.labels = labels
        self.single_class = single_class

    def scale_rbbox(
            self, bboxes: np.ndarray,
            scale_factor_x: float,
            scale_factor_y: float
    ) -> np.ndarray:
        """Scaling rotated boxes.

        :param bboxes: np array of bboxes, shape Nx5. Row is [cx, cy, w, h, angle]
        :param scale_factor_x: scale factor for x coordinates
        :param scale_factor_y: scale factor for y coordinates
        """
        bboxes_zero_angle = bboxes[
            np.any([bboxes[:, 4] == 0, bboxes[:, 4] == 180], axis=0)
        ]
        bboxes_not_zero_angle = bboxes[
            np.all([bboxes[:, 4] != 0, bboxes[:, 4] != 180, bboxes[:, 4] != 90], axis=0)
        ]

        if bboxes_not_zero_angle.shape[0] > 0:
            scale_x = np.array([scale_factor_x] * bboxes_not_zero_angle.shape[0])
            scale_y = np.array([scale_factor_y] * bboxes_not_zero_angle.shape[0])
            scale_x_2 = scale_x * scale_x
            scale_y_2 = scale_y * scale_y
            cotan = 1 / np.tan(bboxes_not_zero_angle[:, 4] / 180 * np.pi)
            cotan_2 = cotan * cotan
            scale_angle = np.arccos(
                scale_x
                * np.sign(bboxes_not_zero_angle[:, 4])
                / np.sqrt(scale_x_2 + scale_y_2 * cotan * cotan)
            )
            nscale_height = np.sqrt(scale_x_2 + scale_y_2 * cotan_2) / np.sqrt(
                1 + cotan_2
            )
            ayh = 1 / np.tan((90 - bboxes_not_zero_angle[:, 4]) / 180 * np.pi)
            nscale_width = np.sqrt(scale_x_2 + scale_y_2 * ayh * ayh) / np.sqrt(
                1 + ayh * ayh
            )
            bboxes_not_zero_angle[:, 4] = 90 - (scale_angle * 180) / np.pi
            bboxes_not_zero_angle[:, 3] = bboxes_not_zero_angle[:, 3] * nscale_height
            bboxes_not_zero_angle[:, 2] = bboxes_not_zero_angle[:, 2] * nscale_width
            bboxes_not_zero_angle[:, 1] = bboxes_not_zero_angle[:, 1] * scale_y
            bboxes_not_zero_angle[:, 0] = bboxes_not_zero_angle[:, 0] * scale_x

        if bboxes_zero_angle.shape[0] > 0:
            bboxes_zero_angle[:, 3] = bboxes_zero_angle[:, 3] * scale_factor_y
            bboxes_zero_angle[:, 2] = bboxes_zero_angle[:, 2] * scale_factor_x
            bboxes_zero_angle[:, 1] = bboxes_zero_angle[:, 1] * scale_factor_y
            bboxes_zero_angle[:, 0] = bboxes_zero_angle[:, 0] * scale_factor_x

        return np.concatenate([bboxes_zero_angle, bboxes_not_zero_angle])

    def predict_on_images(self, ds: ImagesDataset, *args, **kwargs) -> ImagesDataset:
        pred_ds_meta = deepcopy(ds.dataset_meta)
        pred_ds_meta.description = "Prediction: " + pred_ds_meta.description
        pred_ds = ImagesDataset(dataset_meta=pred_ds_meta)
        ds_item: ImageDatasetItem
        for ds_item in tqdm(ds):
            pred_list = self._predict(ds_item.load_image(), ds_item.file_info.unique_id, *args, **kwargs)
            pred_ds.add_item(
                file_info=ds_item.file_info,
                annotation=ImageAnnotation(
                    objects=pred_list
                )
            )
        return pred_ds

    def _predict(self, img: np.ndarray, ref_yolov7_preproc: bool=True) -> List[PredictedObject]:
        if ref_yolov7_preproc:
            h0, w0 = img.shape[:2]  # orig hw
            r = min(self.input_image_size[0] / w0, self.input_image_size[1] / h0)  # ratio
            if r != 1:  # if sizes are not equal
                img_ref_resize = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                                            interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)

            ref_image_shape = img_ref_resize.shape
            img_ref_resize_input, ratio, pad = letterbox(
                img_ref_resize,
                (self.input_image_size[0], self.input_image_size[1]),
                auto=False,
                scaleup=False
            )
        else:
            img_ref_resize_input = cv2.resize(img, self.input_image_size)
            ref_image_shape = img_ref_resize_input.shape

        img_ref_tensor = torch.from_numpy(img_ref_resize_input.transpose((2, 0, 1))).to(self.device).unsqueeze(0)
        img_ref_tensor = img_ref_tensor.half() if self.half else img_ref_tensor.float()  # uint8 to fp16/32
        img_ref_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        pred, train_out = self.model(img_ref_tensor, augment=False)
        del img_ref_tensor
        max_det = 1000
        pred_nms = non_max_suppression_obb(
            prediction=pred,
            conf_threshold=self.det_conf_threshold,
            iou_thres=self.iou_threshold,
            multi_label=False,
            max_det=max_det
        )
        # print("out after non_max_suppression_obb=", pred_nms)

        pred_nms_np = pred_nms[0].detach().cpu().numpy()
        del pred_nms
        pred_nms_np[:, 4] = pred_nms_np[:, 4] / np.pi * 180
        if ref_yolov7_preproc:
            pred_nms_np[:, 0] -= pad[0]
            pred_nms_np[:, 1] -= pad[1]
        pred_obj = []
        ref_height, ref_width = img.shape[:2]
        scale_pred_nms_np = np.concatenate(
            [
                self.scale_rbbox(
                    pred_nms_np[:, :5],
                    ref_width / ref_image_shape[1],
                    ref_height / ref_image_shape[0]
                ),
                pred_nms_np[:, 5:7]
            ],
            axis=1
        )
        for dt in scale_pred_nms_np:
            class_id = dt[-1]
            x, y, w, h, a, conf = [float(t) for t in dt[:-1]]
            pred_o = PredictedObject(
                bbox=RBBoxXYCenterWHA(
                    x_center=x,
                    y_center=y,
                    width=w,
                    height=h,
                    angle=a
                ),
                label=self.labels[class_id] if self.labels else class_id,
                confidence=conf
            )
            pred_obj.append(
                pred_o
            )

        return pred_obj

    def load_model(self, weight_file: Path):
        self.model = attempt_load(weight_file, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half().eval()
