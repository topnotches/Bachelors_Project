import tensorflow as tf
from absl import app, flags, logging
from yolov3_tf2.models import *

import yolov3_tf2.dataset as dataset

from tensorflow_model_optimization.quantization.keras import vitis_quantize

def main(_argv):


    float_model = tf.keras.models.load_model('new_model.h5', custom_objects= {'tf':tf})

    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                            (59, 119), (116, 90), (156, 198), (373, 326), (1,1), (1,1)],
                            np.float32) / 416
    yolo_anchor_masks = np.array([[0,1,2,3,4,5,6,7,8],[9],[10]])

    val_dataset = dataset.load_tfrecord_dataset('./data/voc2012_val.tfrecord', './data/voc2012.names', 224)

    val_dataset = val_dataset.batch(16)
    val_dataset = val_dataset.map(lambda x, y: (dataset.transform_images(x, 224),dataset.transform_targets(y, yolo_anchors, yolo_anchor_masks, 224)))

    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=val_dataset)#, calib_batch_size=16, include_fast_ft=True,fast_ft_epochs=1)
    print(quantized_model.output_shape)
    quantized_model.save('quantized_model.h5')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
