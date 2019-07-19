import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tf.contrib.lite import convert_savedmodel

model_dir = "/media/aswin/336163b5-742f-40f5-9e55-c18e519d29f3/aipc_dataset/model_regress/simple_save/"
out_lite = "/media/aswin/336163b5-742f-40f5-9e55-c18e519d29f3/aipc_dataset/model_regress/lite_save/litemodel.tflite"

convert_savedmodel.convert(saved_model_dir= model_dir, output_tflite= out_lite)

#API
interpreter = tf.contrib.lite.Interpreter(out_lite)
in_tensor = interpreter.get_input_details()[0]["index"]

in_data = np.array([1,2,3])

interpreter.set_tensor(in_tensor, in_data)
interpreter.invoke()
prediction = interpreter.get_tensor(output)
