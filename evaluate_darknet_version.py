import argparse

from absl import flags, app
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from ciou import nmsDiou
from core.config import cfg
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from os.path import join

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#--weights "prova" \ -m "iou"



flags.DEFINE_string('annotation_path', cfg.TEST.ANNOT_PATH, 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')
flags.DEFINE_string('intersectionMethod', "iou", 'write image path')
flags.DEFINE_string('weights', "iou", 'write image path')
flags.DEFINE_string('size','608', 'write image path')
flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3 or yolov3-tiny')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('framework', 'tf', '(tf, tflite')
#flags.DEFINE_float('thresh', 0.0, 'write image path')

one_class=cfg.TEST.Oneclass
print_image=cfg.TEST.PrintImage

"""
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights',  default="none",type=str,help="weights path")
parser.add_argument('-m','--intersectionMethod', default="none",type=str,  help="diou or iou")

args, unknown = parser.parse_known_args()
"""



def main(_argv):
    #cfg.TEST.SCORE_THRESHOLD=FLAGS.thresh
    print(cfg.TEST.SCORE_THRESHOLD.__str__())
    INPUT_SIZE = int(FLAGS.size)
    #cfg.TEST.IntersectionMethod=args.method
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    images_dir_path = './mAP/images'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    #if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)
    #if os.path.exists(cfg.TEST.GT_IMAGE_PATH): shutil.rmtree(cfg.TEST.GT_IMAGE_PATH)
    if os.path.exists(images_dir_path): shutil.rmtree(images_dir_path)

    os.makedirs(predicted_dir_path)
    os.makedirs(ground_truth_dir_path)
    #cfg.TEST.DECTECTED_IMAGE_PATH=cfg.TEST.DECTECTED_IMAGE_PATH +  cfg.TEST.SCORE_THRESHOLD.__str__()
    #cfg.TEST.GT_IMAGE_PATH=cfg.TEST.GT_IMAGE_PATH + cfg.TEST.SCORE_THRESHOLD.__str__()
    if not os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH):
        os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)
    if not os.path.exists(cfg.TEST.GT_IMAGE_PATH):
        os.mkdir(cfg.TEST.GT_IMAGE_PATH)
    os.mkdir(images_dir_path)

    # Build Model
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([int(INPUT_SIZE), int(INPUT_SIZE), 3])
        if FLAGS.tiny:
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_tiny(model, FLAGS.weights)
        else:
            if FLAGS.model == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights_v3(model, FLAGS.weights)
            elif FLAGS.model == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights(model, FLAGS.weights)

    else:
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)

    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            #annotation = line.strip().split()
            annotation=get_bb_list(line)
            image_path = line.replace("\n","")

            image_name = image_path.split('/')[-1]
            shutil.copy(image_path, join(images_dir_path,image_name))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(image_name.split(".")[-2]) + '.txt')
            #print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    #print('\t' + str(bbox_mess).strip())
            #print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(image_name.split(".")[-2]) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if FLAGS.framework == "tf":
                pred_bbox = model.predict(image_data)
            else:
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3':
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
            elif FLAGS.model == 'yolov4':
                XYSCALE = cfg.YOLO.XYSCALE
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=XYSCALE)

            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            if cfg.TEST.IntersectionMethod=="diou":
                bboxes = nmsDiou(bboxes, cfg.TEST.DIOU_NMS_THRESHOLD, method=cfg.TEST.IntersectionMethod)
            else:
                bboxes = utils.nms(bboxes, cfg.TEST.IOU_NMS_THRESHOLD, method="nms")


            if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                image2 = utils.draw_bbox(np.copy(image), bboxes)
                if print_image:
                    cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image2)
                image3= utils.draw_bbox_gt(np.copy(image),annotation)
                if print_image:
                    cv2.imwrite(cfg.TEST.GT_IMAGE_PATH + image_name, image3)


            with open(predict_result_path, 'w') as f:
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                   # class_ind = 0#int(bbox[5])
                    class_ind=int(bbox[5])
                    if one_class:
                        class_ind = 0
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    #print('\t' + str(bbox_mess).strip())
            if num%100==0:
                print(num, num_lines)


def read_annotation(file):
    with open(file, 'r') as annotation_file:
        for  line in enumerate(annotation_file):
            print("dfdf")

def get_bb_list(image_path,training=False):
    if   "combined" not in image_path:
        image = np.array(cv2.imread(join(image_path.replace("\n",""))))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
    else:
        height=2560
        width=1440
    list_bb = []
    with open(image_path.replace(".jpg", ".txt").replace("\n",""), 'r') as f:
        for L in f.readlines():
            row = L.replace("\n", "").split(" ")
            label = row[0]
            if one_class:
                label=0
            centerx = float(row[1]) * width
            centery = float(row[2]) * height
            width_bb = float(row[3]) * width
            height_bb = float(row[4]) * height
            x1 = int(centerx - (width_bb / 2))
            x2 = int(centerx + (width_bb / 2))
            y1 = int(centery - (height_bb / 2))
            y2 = int(centery + (height_bb / 2))
            if not training:
                str = ",".join([x1.__str__(), y1.__str__(), x2.__str__(), y2.__str__(), label.__str__()])
            else:
                str=image_path.replace(".txt",".jpg").replace("\n","")+" "+",".join([x1.__str__(), y1.__str__(), x2.__str__(), y2.__str__(), label.__str__()])
            list_bb.append(str)
    return list_bb


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


