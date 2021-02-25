import shutil

from absl import app, flags, logging
import os
import pickle
from os import listdir
from os.path import isfile, join
from absl.flags import FLAGS
import cv2
import numpy as np
from evaluate_darknet_version import get_bb_list

ValidationPath="data/validation_subset.txt"


def main(_argv):
    with open(ValidationPath, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            # annotation = line.strip().split()
            annotation = get_bb_list(line)
            image_path = line.replace("\n", "")

            image_name = image_path.split('/')[-1]
            #shutil.copy(image_path, join(images_dir_path, image_name))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bblist = [list(map(int, box.split(','))) for box in annotation]
            for bb2 in bblist:
                cv2.rectangle(image, (int(bb2[0]), int(bb2[1])), (int(bb2[2]), int(bb2[3])),(255,0,0), 5)
            image=cv2.resize(image,(600,600))
            cv2.imshow("prova",image)
            cv2.waitKey()




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass