import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

from tensorflow.keras.models import save_model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output', './serving/yolov3/1', 'path to saved_model')
flags.DEFINE_string('classes', './data/voc2012.names', 'path to classes file')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    yolo_net = yolo.get_layer('yolo_darknet')
    print(yolo_net.output_shape)
    save_model(yolo_net, 'new_model.h5', save_format='h5')


    newInput = Input(batch_shape=(1,224,224,3))
    newOutputs = yolo(newInput)
    newModel = Model(newInput,newOutputs)
    tf.saved_model.save(newModel, FLAGS.output)

    #logging.info("model saved to: {}".format(FLAGS.output))

    #model = tf.saved_model.load(FLAGS.output)
    #infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    #logging.info(infer.structured_outputs)
#
    #class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    #logging.info('classes loaded')
#
    #img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    #img = tf.expand_dims(img, 0)
    #img = transform_images(img, 416)
#
    #t1 = time.time()
    #outputs = infer(img)
    #boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
    #    "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
    #t2 = time.time()
    #logging.info('time: {}'.format(t2 - t1))
#
    #logging.info('detections:')
    #for i in range(nums[0]):
    #    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
    #                                       scores[0][i].numpy(),
    #                                       boxes[0][i].numpy()))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass