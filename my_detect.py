import os
import time
from os.path import join

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from PIL import Image
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf

flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', 'data/yolov4-obj_44000.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './screenshot_0.jpg', 'path to input image')
flags.DEFINE_string('output', 'detection', 'path to output image')

def main(_argv):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    input_size = FLAGS.size
    image_path = FLAGS.image
    PATH_rico = "data\\obj\\combined"
    list_jpg_file = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH_rico) for f in filenames if
                     os.path.splitext(f)[1] == '.jpg' and "combined" in os.path.join(dp, f) ]
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])

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

    for image_path in list_jpg_file:
        if list_jpg_file.index(image_path)%1000==0:
            print(list_jpg_file.index(image_path).__str__()+" on " +len(list_jpg_file).__str__())


        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)



        #model.summary()
        pred_bbox = model.predict(image_data)

        if FLAGS.model == 'yolov4':
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
        else:
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_NMS_THRESHOLD, method='nms')

        with open(image_path.replace(".jpg",".txt"), "w") as file1:
            # Writing data to a file


            for f in bboxes:
                width=1440
                height=2560
                x_min=float(f[0])
                y_min=float(f[1])
                x_max=float(f[2])
                y_max=float(f[3])
                index_class=int(f[5])

                width_bb=(x_max-x_min)
                height_bb=(y_max - y_min)
                elem1= ((width_bb/2)+ x_min)/width
                elem2 = ((height_bb/2)+ y_min)/height
                elem3= width_bb/width
                elem4=height_bb/height
                file1.write(index_class.__str__()+" "+elem1.__str__()+" "+elem2.__str__()+" "+elem3.__str__()+" "+elem4.__str__()+ "\n")

        """
        image = utils.draw_bbox(original_image, bboxes)
        image = Image.fromarray(image)
        #image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        if not os.path.exists(FLAGS.output):
            os.makedirs(FLAGS.output)
        cv2.imshow("prova",image)
        cv2.waitKey()
        #cv2.imwrite(join(FLAGS.output,os.path.basename(image_path)), image)
        """

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
