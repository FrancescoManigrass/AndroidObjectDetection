import numpy as np
import tensorflow as tf
import math


from core.config import cfg
def compute_ciou(target,  output):
    '''
    takes in a list of bounding boxes
    but can work for a single bounding box too
    all the boundary cases such as bounding boxes of size 0 are handled.
    ''' 
    target = (target)*(target != 0)
    output = (output)*(target != 0)

    x1g, y1g, x2g, y2g = tf.split(value=target, num_or_size_splits=4, axis=1)
    x1, y1, x2, y2 = tf.split(value=output, num_or_size_splits=4, axis=1)
    
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)
    
    ###iou term###
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)

    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)

    boxAArea = (x2g - x1g +1) * (y2g - y1g +1)
    boxBArea = (x2 - x1 +1) * (y2 - y1 +1)

    iouk = interArea / (boxAArea + boxBArea - interArea + 1e-10)
    ###
    
    ###distance term###
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    ###

    ###aspect-ratio term###
    arctan = tf.atan(w_gt/(h_gt + 1e-10))-tf.atan(w_pred/(h_pred + 1e-10))
    v = (4 / (math.pi ** 2)) * tf.pow((tf.atan(w_gt/(h_gt + 1e-10))-tf.atan(w_pred/(h_pred + 1e-10))),2)
    S = 1 - iouk
    alpha = v / (S + v + 1e-10)
    w_temp = 2 * w_pred
    ar = (8 / (math.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)
    ###
    
    ###calculate diou###
    #diouk = iouk - u
    #diouk = (1 - diouk)
    ###
    
    ###calculate ciou###
    ciouk = iouk - (u + alpha * ar)
    ciouk = (1 - ciouk)
    ###
    
    return ciouk


def computerIOU(target,output):
    target = (target) * (target != 0)
    output = (output) * (target != 0)

    x1g, y1g, x2g, y2g = target[0], target[1], target[2], target[3]
    x1, y1, x2, y2 = output[0], output[1], output[2], output[3]

    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)

    ###iou term###
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)

    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)

    boxAArea = (x2g - x1g + 1) * (y2g - y1g + 1)
    boxBArea = (x2 - x1 + 1) * (y2 - y1 + 1)

    iouk = interArea / (boxAArea + boxBArea - interArea + 1e-10)


    return iouk


def computeDiou(target,  output):
    '''
    takes in a list of bounding boxes
    but can work for a single bounding box too
    all the boundary cases such as bounding boxes of size 0 are handled.
    '''
    target = (target) * (target != 0)
    output = (output) * (target != 0)

    x1g, y1g, x2g, y2g = target[0],target[1],target[2],target[3]
    x1, y1, x2, y2 = output[0],output[1],output[2],output[3]

    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2



    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)


    ###iou term###
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)

    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)

    boxAArea = (x2g - x1g + 1) * (y2g - y1g + 1)
    boxBArea = (x2 - x1 + 1) * (y2 - y1 + 1)

    iouk = interArea / (boxAArea + boxBArea - interArea + 1e-10)
    ###


    ###distance term###
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    ###

    ###aspect-ratio term###

    arctan = tf.atan(w_gt / (h_gt + 1e-10)) - tf.atan(w_pred / (h_pred + 1e-10))
    v = (4 / (math.pi ** 2)) * tf.pow((tf.atan(w_gt / (h_gt + 1e-10)) - tf.atan(w_pred / (h_pred + 1e-10))), 2)
    S = 1 - iouk
    alpha = v / (S + v + 1e-10)
    w_temp = 2 * w_pred
    ar = (8 / (math.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)


    ###

    ###calculate diou###
    diouk = iouk - cfg.TEST.DIOU_LAMBDA*u #+alpha * ar)
    #diouk = (1 - diouk)
    ###

    ###calculate ciou###
    #ciouk = iouk - (u + alpha * ar)
    #ciouk = (1 - ciouk)
    ###

    return diouk


def nmsDiou(bboxes, diou_threshold, sigma=0.3, method='diou'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            Diou = bboxes_diou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(Diou),), dtype=np.float32)

            assert method in ['diou']

            if method == 'diou':


                iou_mask = Diou > diou_threshold
                weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes



def bboxes_diou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    '''

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        '''
    '''
        takes in a list of bounding boxes
        but can work for a single bounding box too
        all the boundary cases such as bounding boxes of size 0 are handled.
        '''
    #target = (target) * (target != 0)
    #output = (output) * (target != 0)
    #boxes1= target1
    #boxes2= output

    x1g, y1g, x2g, y2g = boxes1[..., 0], boxes1[..., 1], boxes1[..., 2], boxes1[..., 3]
    x1, y1, x2, y2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]

    w_pred = np.subtract(x2, x1)
    h_pred = np.subtract(y2,y1)
    w_gt = np.subtract(x2g, x1g)
    h_gt = np.subtract(y2g, y1g)

    x_center = np.divide(np.add(x2 , x1) , 2)
    y_center = np.divide(np.add(y2 , y1) , 2)
    x_center_g = np.divide(np.add(x1g , x2g) , 2)
    y_center_g = np.divide(np.add(y1g , y2g) , 2)

    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)

    ###iou term###
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)

    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)

    boxAArea = (x2g - x1g + 1) * (y2g - y1g + 1)
    boxBArea = (x2 - x1 + 1) * (y2 - y1 + 1)

    iouk = interArea / (boxAArea + boxBArea - interArea + 1e-10)
    ###

    ###distance term###
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    ###

    ###aspect-ratio term###
    arctan = tf.atan(w_gt / (h_gt + 1e-10)) - tf.atan(w_pred / (h_pred + 1e-10))
    v = (4 / (math.pi ** 2)) * tf.pow((tf.atan(w_gt / (h_gt + 1e-10)) - tf.atan(w_pred / (h_pred + 1e-10))), 2)
    S = 1 - iouk
    alpha = v / (S + v + 1e-10)
    w_temp = 2 * w_pred
    ar = (8 / (math.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)

    #diouk = np.array(u)
    ###

    ###calculate diou###
    diouk = iouk - cfg.TEST.DIOU_LAMBDA*u
    # diouk = (1 - diouk)
    ###
    diouk = np.array(diouk)

    ###calculate ciou###
    # ciouk = iouk - (u + alpha * ar)
    # ciouk = (1 - ciouk)
    ###

    return diouk
