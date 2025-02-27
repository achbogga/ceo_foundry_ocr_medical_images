# Copyright 2019 Achyut Sarma Boggaram (achbogga@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL VOC annotations to TFRecord for object_detection.

Example usage in colab:
    !python 'drive/My Drive/CEO_Foundry_LLC/create_tf_records_from_pascal_voc_annotations_achbogga.py' \
    --label_map_path='drive/My Drive/CEO_Foundry_LLC/medical_images_label_map.pbtxt' \
    --image_data_dir='drive/My Drive/CEO_Foundry_LLC/Total' \
    --annotations_dir='drive/My Drive/CEO_Foundry_LLC/pascal_voc_output_training' \
    --output_path='drive/My Drive/CEO_Foundry_LLC/medical_images.record'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('image_data_dir', '', 'Path to the Directory containing all the iamges')
flags.DEFINE_string('annotations_dir', '',
                    'Path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS



def dict_to_tf_example(data,
                       image_data_directory,
                       label_map_dict,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  
  full_path = os.path.join(image_data_directory, data['filename'])
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG' and image.format != 'PNG':
    raise ValueError('Image format not JPEG neither PNG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      if obj['difficult']=='Unspecified':
        continue
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):

  image_data_dir = FLAGS.image_data_dir
  annotations_dir = FLAGS.annotations_dir
  
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  
  logging.info('Reading from %s dataset.', image_data_dir)
  files_list = os.listdir(annotations_dir)
  examples_list = files_list
  #print (len(examples_list))
  for idx, example in enumerate(examples_list):
    #print (example)
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples_list))
    path = os.path.join(annotations_dir, example)
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, FLAGS.image_data_dir, label_map_dict,
                                    FLAGS.ignore_difficult_instances)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()