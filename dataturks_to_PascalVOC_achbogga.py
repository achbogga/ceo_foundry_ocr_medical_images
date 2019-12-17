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

r"""Convert dataturks annotations json to PASCAL_VOC Xmls for object_detection.

Example usage in colab:
	!python "drive/My Drive/CEO_Foundry_LLC/dataturks_to_PascalVOC_achbogga.py" --dataturks_JSON_FilePath "/content/drive/My Drive/CEO_Foundry_LLC/Medical Images.json" \
    --image_download_dir "/content/drive/My Drive/CEO_Foundry_LLC/Total" \
    --pascal_voc_xml_train_dir "/content/drive/My Drive/CEO_Foundry_LLC/pascal_voc_output_training" \
    --pascal_voc_xml_validation_dir "/content/drive/My Drive/CEO_Foundry_LLC/pascal_voc_output_validation" \
    --validation_split 0.2
"""

import argparse
import sys
import os
import json
import logging
from PIL import Image
import random

###################  INSTALLATION NOTE #######################
##############################################################

## pip install pillow

###############################################################
###############################################################


#enable info logging.
logging.getLogger().setLevel(logging.INFO)


def get_xml_for_bbx(bbx_label, bbx_data, width, height):

	if len(bbx_data['points']) == 4:
		#Regular BBX has 4 points of the rectangle.
		xmin = width*min(bbx_data['points'][0][0], bbx_data['points'][1][0], bbx_data['points'][2][0], bbx_data['points'][3][0])
		ymin = height * min(bbx_data['points'][0][1], bbx_data['points'][1][1], bbx_data['points'][2][1],
						   bbx_data['points'][3][1])

		xmax = width * max(bbx_data['points'][0][0], bbx_data['points'][1][0], bbx_data['points'][2][0],
						   bbx_data['points'][3][0])
		ymax = height * max(bbx_data['points'][0][1], bbx_data['points'][1][1], bbx_data['points'][2][1],
						   bbx_data['points'][3][1])

	else:
		#OCR BBX format has 'x','y' in one point.
		# We store the left top and right bottom as point '0' and point '1'
		xmin = int(bbx_data['points'][0]['x']*width)
		ymin = int(bbx_data['points'][0]['y']*height)
		xmax = int(bbx_data['points'][1]['x']*width)
		ymax = int(bbx_data['points'][1]['y']*height)

	xml = "<object>\n"
	xml = xml + "\t<name>" + bbx_label + "</name>\n"
	xml = xml + "\t<pose>Unspecified</pose>\n"
	xml = xml + "\t<truncated>Unspecified</truncated>\n"
	xml = xml + "\t<difficult>Unspecified</difficult>\n"
	xml = xml + "\t<occluded>Unspecified</occluded>\n"
	xml = xml + "\t<bndbox>\n"
	xml = xml +     "\t\t<xmin>" + str(xmin) + "</xmin>\n"
	xml = xml +     "\t\t<xmax>" + str(xmax) + "</xmax>\n"
	xml = xml +     "\t\t<ymin>" + str(ymin) + "</ymin>\n"
	xml = xml +     "\t\t<ymax>" + str(ymax) + "</ymax>\n"
	xml = xml + "\t</bndbox>\n"
	xml = xml + "</object>\n"
	return xml


def convert_to_PascalVOC(dataturks_labeled_item, image_dir, xml_out_dir):

	"""Convert a dataturks labeled item to pascalVOCXML string.
	  Args:
		dataturks_labeled_item: JSON of one labeled image from dataturks.
		image_dir: Path to  directory to downloaded images (or a directory already having the images downloaded).
		xml_out_dir: Path to the dir where the xml needs to be written.
	  Returns:
		None.
	  Raises:
		None.
	  """
	try:
		data = json.loads(dataturks_labeled_item)
		if len(data['annotation']) == 0:
			logging.info("Ignoring Skipped Item");
			return False;

		width = data['annotation'][0]['imageWidth']
		height = data['annotation'][0]['imageHeight']
		image_url = data['content']

		#adding adhoc fix for now
		image_url_new = image_url.replace(image_url.split('___Total_')[0]+'___Total_', '')

		filePath = image_dir+'/'+image_url_new
		print (filePath)

		with Image.open(filePath) as img:
			width, height = img.size

		fileName = filePath.split("/")[-1]
		image_dir_folder_Name = image_dir.split("/")[-1]


		xml = "<annotation>\n<folder>" + image_dir_folder_Name + "</folder>\n"
		xml = xml + "<filename>" + fileName +"</filename>\n"
		xml = xml + "<path>" + filePath +"</path>\n"
		xml = xml + "<source>\n\t<database>Unknown</database>\n</source>\n"
		xml = xml + "<size>\n"
		xml = xml +     "\t<width>" + str(width) + "</width>\n"
		xml = xml +    "\t<height>" + str(height) + "</height>\n"
		xml = xml +    "\t<depth>Unspecified</depth>\n"
		xml = xml +  "</size>\n"
		xml = xml + "<segmented>Unspecified</segmented>\n"

		for bbx in data['annotation']:
			if not bbx:
				continue;
			#Pascal VOC only supports rectangles.
			if "shape" in bbx and bbx["shape"] != "rectangle":
				continue;

			bbx_labels = bbx['label']
			#handle both list of labels or a single label.
			if not isinstance(bbx_labels, list):
				bbx_labels = [bbx_labels]

			for bbx_label in bbx_labels:
				xml = xml + get_xml_for_bbx(bbx_label, bbx, width, height)

		xml = xml + "</annotation>"

		#output to a file.
		xmlFilePath = os.path.join(xml_out_dir, fileName + ".xml")
		with open(xmlFilePath, 'w') as f:
			f.write(xml)
		return True
	except Exception as e:
		logging.exception("Unable to process item " + dataturks_labeled_item + "\n" + "error = "  + str(e))
		return False

def main(dataturks_JSON_FilePath, image_download_dir, pascal_voc_xml_train_dir, pascal_voc_xml_validation_dir, validation_split):
	#make sure everything is setup.
	if (not os.path.isdir(image_download_dir)):
		logging.exception("Please specify a valid directory path to download images, " + image_download_dir + " doesn't exist")
		return
	if (not os.path.isdir(pascal_voc_xml_train_dir)):
		os.makedirs(pascal_voc_xml_train_dir)
	if (not os.path.isdir(pascal_voc_xml_validation_dir)):
		os.makedirs(pascal_voc_xml_validation_dir)
	if (not os.path.exists(dataturks_JSON_FilePath)):
		logging.exception(
			"Please specify a valid path to dataturks JSON output file, " + dataturks_JSON_FilePath + " doesn't exist")
		return

	lines = []
	with open(dataturks_JSON_FilePath, 'r') as f:
		lines = f.readlines()

	if (not lines or len(lines) == 0):
		logging.exception(
			"Please specify a valid path to dataturks JSON output file, " + dataturks_JSON_FilePath + " is empty")
		return

	validation_indices = list(random.sample(range(len(lines)), int(len(lines)*validation_split)))
	success = 0
	for idx, line in enumerate(lines):
		if idx not in validation_indices:
			status = convert_to_PascalVOC(line, image_download_dir, pascal_voc_xml_train_dir)
		else:
			status = convert_to_PascalVOC(line, image_download_dir, pascal_voc_xml_validation_dir)
		if (status):
			success = success + 1
		if (idx % 10 == 0):
			logging.info(str(idx) + " items done ...")

	logging.info("Completed: " + str(success) + " items done, " + str(len(lines) - success)  + " items ignored due to errors or for being skipped items.")


def create_arg_parser():
	""""Creates and returns the ArgumentParser object."""

	parser = argparse.ArgumentParser(description='Converts Dataturks output JSON file for Image bounding box to Pascal VOC format.')
	parser.add_argument('--dataturks_JSON_FilePath',
					help='Path to the JSON file downloaded from Dataturks.')
	parser.add_argument('--image_download_dir',
					help='Path to the directory where images are dowloaded')
	parser.add_argument('--pascal_voc_xml_train_dir',
						help='Path to the directory where the training split Pascal VOC XML files will be stored.')
	parser.add_argument('--pascal_voc_xml_validation_dir',
						help='Path to the directory where the validation split Pascal VOC XML files will be stored.')
	parser.add_argument('--validation_split',
						type = float,
						help='The validation split percentage in the range of 0.0 to 1.0', default = 0.2)
	return parser

if __name__ == '__main__':
	arg_parser = create_arg_parser()
	parsed_args = arg_parser.parse_args(sys.argv[1:])
	main(parsed_args.dataturks_JSON_FilePath, parsed_args.image_download_dir, parsed_args.pascal_voc_xml_train_dir, parsed_args.pascal_voc_xml_validation_dir, parsed_args.validation_split)