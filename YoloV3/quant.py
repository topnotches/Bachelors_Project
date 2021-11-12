import tensorflow as tf
from absl import app, flags, logging
from yolov3_tf2.models import *

import yolov3_tf2.dataset as dataset

from tensorflow_model_optimization.quantization.keras import vitis_quantize

def main(_argv):


    float_model = tf.keras.models.load_model('new_model.h5', custom_objects= {'tf':tf})

    yolo_anchors = np.array([(0.23551913026451718, 0.32283159910773246), (0.3301928883940777, 0.487984731908737), (0.18882981712257776, 0.20363369724054445), (0.07460299343068891, 0.09748286841560129), (0.001, 0.001), (0.001, 0.001)],
                            np.float32)
    yolo_anchor_masks = np.array([[0,1,2,3],[4],[5]])
    val_dataset = dataset.load_tfrecord_dataset('./data/voc2012_val.tfrecord', './data/voc2012.names', 224)

    val_dataset = val_dataset.batch(4)
    val_dataset = val_dataset.map(lambda x, y: (dataset.transform_images(x, 224), dataset.transform_targets(y, yolo_anchors, yolo_anchor_masks, 224)))

    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=val_dataset, calib_batch_size=16, include_fast_ft=True, fast_ft_epochs=15)
    print(quantized_model.output_shape)
    quantized_model.save('quantized_model.h5')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
