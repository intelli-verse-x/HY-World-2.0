from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from skimage import morphology
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


def build_gd_model(GROUNDING_MODEL, device="cuda"):
    # build grounding dino from huggingface
    model_id = GROUNDING_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return processor, grounding_model


def build_box_segmenter(checkpoint, device="cuda"):
    """Build the Apache-2.0 SAM box-prompt segmenter."""
    processor = SamProcessor.from_pretrained(checkpoint)
    model = SamModel.from_pretrained(checkpoint).to(device).eval()
    return BoxSegmenter(model, processor, device)


class BoxSegmenter:
    """Small predictor adapter matching the box-mask contract used by WorldNav."""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.image = None

    def set_image(self, image):
        self.image = image

    def predict(
            self,
            point_coords: [np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            box: Optional[np.ndarray] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if self.image is None:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            raise NotImplementedError("WorldNav's licensed segmenter adapter accepts box prompts only")
        if box is None:
            raise ValueError("At least one box prompt is required")

        boxes = np.asarray(box, dtype=np.float32).reshape(-1, 4)
        inputs = self.processor(
            images=self.image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        if masks.ndim == 4:
            masks = masks[:, 0]
        masks_np = masks.float().numpy()
        if not return_logits:
            masks_np = masks_np > 0.0

        scores = outputs.iou_scores[0]
        if scores.ndim == 2:
            scores = scores[:, 0]
        low_res_masks = outputs.pred_masks[0]
        if low_res_masks.ndim == 4:
            low_res_masks = low_res_masks[:, 0]
        return (
            masks_np,
            scores.float().detach().cpu().numpy(),
            low_res_masks.float().detach().cpu().numpy(),
        )


# filter the small bboxes to avoid memory overflow
def filter_small_bboxes(results):
    max_num = 100
    bboxes = results[0]["boxes"]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep = [order[i].item() for i in range(min(max_num, order.numel()))]
    return torch.LongTensor(keep)


def get_contours_sky(mask):
    binary = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask

    mask = np.zeros_like(binary)

    cv2.drawContours(mask, contours, -1, 1, -1)

    return mask.astype(np.bool_)


def remove_sky_floaters(mask, min_size=1000):
    mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=2)

    return mask


def get_sky(image, mask_predictor, processor, grounding_model, DEVICE="cuda"):
    text = "sky."
    H, W = image.height, image.width
    mask_predictor.set_image(np.array(image.convert("RGB")))

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    # Send Grounding DINO's boxes to the promptable mask model.
    results[0]["boxes"] = results[0]["boxes"]
    # filter the small boxes to avoid memory overflow
    filter_keep = filter_small_bboxes(results)
    results[0]["boxes"] = results[0]["boxes"][filter_keep]
    results[0]["scores"] = results[0]["scores"][filter_keep]
    results[0]["labels"] = [results[0]["labels"][i] for i in filter_keep]
    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] == 0:
        sky_mask = np.zeros((H, W), dtype=np.bool_)
        return sky_mask

    masks, scores, logits = mask_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    sky_mask = np.zeros((H, W), dtype=np.bool_)

    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.bool_)
        sky_mask[mask] = 1

    # remove the small objects in masks
    min_floater = 500
    sky_mask = sky_mask.astype(np.bool_)
    sky_mask = get_contours_sky(sky_mask)
    sky_mask = 1 - sky_mask  # invert the mask to get the sky area
    sky_mask = sky_mask.astype(np.bool_)
    sky_mask = remove_sky_floaters(sky_mask, min_size=min_floater)
    sky_mask = get_contours_sky(sky_mask)

    return sky_mask


def get_segment_mask(image, text, box_conf, text_conf, mask_predictor, processor, grounding_model, DEVICE="cuda"):
    H, W = image.height, image.width
    mask_predictor.set_image(np.array(image.convert("RGB")))

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_conf,
        text_threshold=text_conf,
        target_sizes=[image.size[::-1]]
    )
    # filter the small boxes to avoid memory overflow
    filter_keep = filter_small_bboxes(results)
    results[0]["boxes"] = results[0]["boxes"][filter_keep]
    # results[0]["scores"] = results[0]["scores"][filter_keep]
    # results[0]["labels"] = [results[0]["labels"][i] for i in filter_keep]
    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] == 0:
        result_mask = np.zeros((H, W), dtype=np.bool_)
        return result_mask

    masks, scores, logits = mask_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    masks = np.clip(np.sum(masks, axis=0, keepdims=True), 0, 1)

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    result_mask = np.zeros((H, W), dtype=np.bool_)

    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.bool_)
        result_mask[mask] = 1

    if type(text) == str and "sky" in text:
        # remove the small objects in masks
        min_floater = 500
        result_mask = result_mask.astype(np.bool_)
        result_mask = get_contours_sky(result_mask)
        result_mask = 1 - result_mask  # invert the mask to get the sky area
        result_mask = result_mask.astype(np.bool_)
        result_mask = remove_sky_floaters(result_mask, min_size=min_floater)
        result_mask = get_contours_sky(result_mask)
    else:
        result_mask = 1 - result_mask
        result_mask = result_mask.astype(np.bool_)

    return result_mask
