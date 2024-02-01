import numpy as np
import torch
from shapely.geometry import Polygon


def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 8, 3) in the form of (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
                                    0       1       2       3       4       5       6       7
    Returns
    -------
        list of converted shapely.geometry.Polygon object.
    """
    # polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    z0_indices = [0, 3, 7, 4]
    polygons = [
        Polygon([(box[i, 0], box[i, 1]) for i in z0_indices])
        for box in boxes_array
    ]
    return np.array(polygons)


def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]
    return np.array(iou, dtype=np.float32)


def calculate_tp_fp(
        det_boxes,
        det_score,
        gt_boxes,
        result_stat,
        iou_thresh,
        left_range=-float('inf'),
        right_range=float('inf'),
):
    """
    Calculate the true positive and false positive numbers of the current frame.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each predict bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    right_range : float
        The evaluation range right bound
    left_range : float
        The evaluation range left bound
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []

    if det_boxes is not None:
        # convert bounding boxes to a numpy array
        det_boxes = det_boxes.detach().cpu().numpy()
        det_score = det_score.detach().cpu().numpy()
        gt_boxes = gt_boxes.detach().cpu().numpy()

        det_polygon_list_origin = list(convert_format(det_boxes))
        gt_polygon_list_origin = list(convert_format(gt_boxes))
        det_polygon_list = []
        gt_polygon_list = []
        det_score_new = []
        # remove out-of-range predicted bbx
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x**2 +
                               det_polygon.centroid.y**2)
            if left_range < distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x**2 +
                               gt_polygon.centroid.y**2)
            if left_range < distance < right_range:
                gt_polygon_list.append(gt_polygon)

        gt = len(gt_polygon_list)
        det_score_new = np.array(det_score_new)
        # sort the predicted bounding boxes by score
        score_order_descend = np.argsort(-det_score_new)

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)
            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
    else:
        gt = gt_boxes.shape[0]
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def voc_ap(rec, prec):
    """VOC 2010 Average Precision."""
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def calculate_ap(result_stat, iou):
    """Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou: float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, range: str = ''):
    ap_dict = {}
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    prefix = (range + '_') if range else ''
    ap_dict.update({
        prefix + 'ap_50': ap_50,
        prefix + 'ap_70': ap_70,
    })
    return ap_dict

