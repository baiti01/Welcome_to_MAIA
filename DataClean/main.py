#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 3/12/2021 12:27 PM

# sys
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import logging
import pydicom
import numpy as np
import nrrd
import cv2
from scipy.ndimage import zoom

# project
from mask_visualization import convert_to_RGB255, apply_mask

# set up logging to file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='log.txt',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

byte_depth_table = {1: np.int8,
                    2: np.int16,
                    4: np.int32,
                    8: np.int64}


def get_study_information(study_folder):
    current_study_data = {}
    for roots, dirs, files in os.walk(study_folder):
        if len(dirs) == 0:
            current_data = pydicom.read_file(os.path.join(roots, files[0]))
            modality = current_data.Modality
            if modality.upper() == 'CT':
                current_study_data['CT'] = roots
            else:
                current_study_data[modality] = os.path.join(roots, files[0])
    return current_study_data


def extract_data_list(data_path):
    studies = os.listdir(data_path)
    data_list = {}
    logger.info('Extracting data list ...')
    for current_study in studies:
        logger.info('Extracting data list: {}'.format(current_study))
        current_study_folder = os.listdir(os.path.join(data_path, current_study))
        for current_sub_folder in current_study_folder:
            if 'simulation' not in current_sub_folder.lower():
                continue
            data_list[current_study] = get_study_information(os.path.join(data_path, current_study, current_sub_folder))
            break
    return data_list


def extract_CT(CT_path):
    dicom_meta_dictionary = {}
    for current_file in os.listdir(CT_path):
        current_file = pydicom.read_file(os.path.join(CT_path, current_file))

        current_image_position = current_file.ImagePositionPatient
        current_z_location = current_image_position[2]

        intercept = np.float(current_file.RescaleIntercept)
        slop = np.float32(current_file.RescaleSlope)

        current_slice_thickness = np.float32(current_file.SliceThickness)
        current_slice_spacing = [np.float32(x) for x in current_file.PixelSpacing]

        current_pixel_data = current_file.PixelData
        current_byte_depth = len(current_pixel_data) // (current_file.Rows * current_file.Columns)
        current_pixel_data = np.copy(np.reshape(np.frombuffer(current_pixel_data,
                                                              dtype=byte_depth_table[current_byte_depth],
                                                              count=current_file.Rows * current_file.Columns),
                                                (current_file.Rows, current_file.Columns))).astype(np.float32)
        current_pixel_data = current_pixel_data * slop + intercept

        dicom_meta_dictionary[current_z_location] = {}
        dicom_meta_dictionary[current_z_location]['ImagePositionPatient'] = current_image_position
        dicom_meta_dictionary[current_z_location]['SliceThickness'] = current_slice_thickness
        dicom_meta_dictionary[current_z_location]['PixelSpacing'] = current_slice_spacing
        dicom_meta_dictionary[current_z_location]['PixelData'] = current_pixel_data
        dicom_meta_dictionary[current_z_location]['ByteDepth'] = np.float32

    # generate CT numpy array
    slice_thickness = [v['SliceThickness'] for k, v in dicom_meta_dictionary.items()]
    pixel_spacing_x = [v['PixelSpacing'][0] for k, v in dicom_meta_dictionary.items()]
    pixel_spacing_y = [v['PixelSpacing'][1] for k, v in dicom_meta_dictionary.items()]

    slice_thickness = list(set(slice_thickness))
    pixel_spacing_x = list(set(pixel_spacing_x))
    pixel_spacing_y = list(set(pixel_spacing_y))

    if len(slice_thickness) != 1:
        logger.info(f'Warning! The number of the slice_thickness does not equal to 1!'
                    f' Data path: {CT_path}')

    if len(pixel_spacing_x) != 1 or len(pixel_spacing_y) != 1:
        logger.info(f'Warning! The number of the pixel_spacing does not equal to 1!'
                    f' Data path: {CT_path}')

    slice_thickness = slice_thickness[0]
    pixel_spacing = (pixel_spacing_x[0], pixel_spacing_y[0])

    # get z-position
    z_positions = [k for k, v in dicom_meta_dictionary.items()]
    sorted_z_positions = sorted(z_positions)
    slice_thickness_list = [sorted_z_positions[i] - sorted_z_positions[i - 1] for i in
                            range(1, len(sorted_z_positions))]
    if 'nan' in str(slice_thickness):
        slice_thickness = np.mean(slice_thickness_list)
        logger.info('Using averaged slice thickness {}'.format(slice_thickness))
    pixel_data = [dicom_meta_dictionary[current_z_location]['PixelData'] for current_z_location in sorted_z_positions]
    pixel_data = np.stack(pixel_data, axis=0)

    pixel_data = (pixel_data + 1000)
    pixel_data[pixel_data < 0] = 0

    current_study = {}
    current_study['CT'] = pixel_data.astype(np.uint16)
    current_study['z_positions'] = sorted_z_positions
    current_study['pixel_spacing'] = pixel_spacing
    current_study['slice_thickness'] = slice_thickness
    current_study['offset'] = {'start': dicom_meta_dictionary[sorted_z_positions[0]]['ImagePositionPatient'],
                               'end': dicom_meta_dictionary[sorted_z_positions[-1]]['ImagePositionPatient']}

    return current_study


def extract_Contours(RTStruct_path, current_study, is_mask_fill=True):
    RTStruct_data = pydicom.read_file(RTStruct_path)
    ROI_names = [x.ROIName for x in RTStruct_data.StructureSetROISequence]
    contour_sequences = RTStruct_data.ROIContourSequence

    fill_flag = -1 if is_mask_fill else 3
    slice_num, rows, columns = current_study['CT'].shape
    current_study['contours'] = {}
    for current_roi_name, current_contour_sequences in zip(ROI_names, contour_sequences):
        mask_volume = np.zeros((slice_num, rows, columns), dtype=np.float32)
        if not hasattr(current_contour_sequences, 'ContourSequence'):
            continue
        for current_contour in current_contour_sequences.ContourSequence:
            if not hasattr(current_contour, 'NumberOfContourPoints'):
                continue
            if not hasattr(current_contour, 'ContourData'):
                continue

            # get contour
            points_number = current_contour.NumberOfContourPoints
            current_contour_data = current_contour.ContourData
            z_location = current_contour_data[2]

            current_contour_data = [[[int((current_contour_data[j * 3 + 0] - current_study['offset']['start'][0]) /
                                          current_study['pixel_spacing'][0]),
                                      int((current_contour_data[j * 3 + 1] - current_study['offset']['start'][1]) /
                                          current_study['pixel_spacing'][1])] for j in
                                     range(points_number)]]
            current_contour_data = np.asarray(current_contour_data)

            # convert to coordinates
            # current_contour_data[:, 0] = (current_contour_data[:, 0] - patient_CT['offset']['start'][0]) / patient_CT['pixel_spacing'][0]
            # current_contour_data[:, 1] = (current_contour_data[:, 1] - patient_CT['offset']['start'][1]) / patient_CT['pixel_spacing'][1]

            # find the slice index
            current_slice_index = np.argmin(
                np.abs(z_location - np.array(np.array(current_study['z_positions']))))

            # generate the mask
            current_slice_mask = cv2.drawContours(np.zeros((rows, columns), dtype=np.float32),
                                                  current_contour_data, -1, 1.0, fill_flag)
            mask_volume[current_slice_index, :, :] += current_slice_mask

        mask_volume[mask_volume >= 0.5] = 1.0
        mask_volume[mask_volume < 0.5] = 0
        current_study['contours'][current_roi_name] = mask_volume.astype(np.uint8)
    return current_study


def save_tmp_data(current_contours, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    header = {'dimension': 3,
              'sizes': current_contours['CT'].shape,
              'spacing': current_contours['pixel_spacing'],
              'slice_thickness': current_contours['slice_thickness'],
              'CT_type': 'uint16',
              'mask_type': 'uint8'}

    # save CT
    nrrd.write(os.path.join(output_path, 'CT.nrrd'), current_contours['CT'], header)

    # save contours
    for k, v in current_contours['contours'].items():
        name = ''.join(e for e in k if (e.isalnum() or e == ' ' or e == '_'))
        nrrd.write(os.path.join(output_path, '{}.nrrd'.format(name)), current_contours['contours'][k], header)


def organ_name_rules(data_path):
    organ_name_dict = {
        'Brainstem': [],
        'Lt_ParotidGland': [],
        'Rt_ParotidGland': [],
        'Lt_ParotidGland_Sup': [],
        'Rt_ParotidGland_Sup': [],
        'Mandible': [],
        'SpinalCord': [],
        'Lt_SubMandibular': [],
        'Rt_SubMandibular': [],
        'Larynx': [],
        'Lt_Lung': [],
        'Rt_Lung': [],
        'Lt_Lens': [],
        'Rt_Lens': [],
        'Lt_Eye': [],
        'Rt_Eye': [],
        'Lt_Cochlea': [],
        'Rt_Cochlea': [],
        'Lt_OpticalNerve': [],
        'Rt_OpticalNerve': [],
        'OpticalChiasma': [],
        'OralCavity': []
    }

    for roots, dirs, files in os.walk(data_path):
        for current_file in files:
            current_file_path = os.path.join(roots, current_file)
            current_file = current_file.lower()
            current_file = os.path.splitext(current_file)[0]
            if 'brainstem' in current_file.lower() and 'exp' not in current_file.lower():
                organ_name_dict['Brainstem'].append((current_file, current_file_path))

            if 'parotid' in current_file.lower():
                if current_file.lower().startswith('l') or current_file.endswith('lt') or current_file.endswith(
                        'left') or current_file.endswith('l') or current_file.endswith('lt') or current_file.endswith(
                    'lt sub'):
                    if 'sub' not in current_file and 'sud' not in current_file:
                        # lt_parotid.append(current_file)
                        organ_name_dict['Lt_ParotidGland'].append((current_file, current_file_path))
                    else:
                        # lt_parotid_sub.append(current_file)
                        organ_name_dict['Lt_ParotidGland_Sup'].append((current_file, current_file_path))

                elif current_file.lower().startswith('r') or current_file.endswith('rt') or current_file.endswith(
                        'right') or current_file.endswith('r') or current_file.endswith('rt'):
                    if 'sub' not in current_file:
                        # rt_parotid.append(current_file)
                        organ_name_dict['Rt_ParotidGland'].append((current_file, current_file_path))
                    else:
                        # rt_parotid_sub.append(current_file)
                        organ_name_dict['Rt_ParotidGland_Sup'].append((current_file, current_file_path))
                else:
                    logger.info('Unused organ name: {}'.format(current_file))

            if 'mandible' in current_file.lower() and 'sub' not in current_file.lower():
                # mandible.append(current_file)
                organ_name_dict['Mandible'].append((current_file, current_file_path))

            if 'spinal' in current_file.lower() and 'cord' in current_file.lower() and 'exp' not in current_file.lower():
                # spinalcord.append(current_file)
                organ_name_dict['SpinalCord'].append((current_file, current_file_path))

            if 'submandibular' in current_file.lower():
                if current_file.lower().startswith('l') or current_file.endswith('lt') or current_file.endswith('left'):
                    # lt_submandibular.append(current_file)
                    organ_name_dict['Lt_SubMandibular'].append((current_file, current_file_path))
                elif current_file.lower().startswith('r') or current_file.endswith('rt') or current_file.endswith(
                        'right'):
                    # rt_submandibular.append(current_file)
                    organ_name_dict['Rt_SubMandibular'].append((current_file, current_file_path))

            if 'larynx' in current_file.lower():
                # larynx.append(current_file)
                organ_name_dict['Larynx'].append((current_file, current_file_path))

            if 'lung' in current_file.lower():
                if current_file.lower().replace('lung', '').startswith('l') or current_file.endswith(
                        'lt') or current_file.endswith('left'):
                    # lt_Lung.append(current_file)
                    organ_name_dict['Lt_Lung'].append((current_file, current_file_path))
                elif current_file.lower().replace('lung', '').startswith('r') or current_file.endswith(
                        'rt') or current_file.endswith('right'):
                    # rt_Lung.append(current_file)
                    organ_name_dict['Rt_Lung'].append((current_file, current_file_path))
                else:
                    logger.info('Unused organ name: {}'.format(current_file))

            if 'lens' in current_file.lower():
                if current_file.lower().startswith('l') or current_file.endswith('lt') or current_file.endswith('left'):
                    if len(current_file) > len(
                            'lens') and 'exp' not in current_file.lower() and 'ltrt' not in current_file.lower():
                        # Lt_Lens.append(current_file)
                        organ_name_dict['Lt_Lens'].append((current_file, current_file_path))
                    else:
                        logger.info('Unused organ name: {}'.format(current_file))
                elif current_file.lower().startswith('r') or current_file.endswith('rt') or current_file.endswith(
                        'right'):
                    if len(current_file) > len(
                            'lens') and 'exp' not in current_file.lower() and 'ltrt' not in current_file.lower():
                        # Rt_Lens.append(current_file)
                        organ_name_dict['Rt_Lens'].append((current_file, current_file_path))
                    else:
                        logger.info('Unused organ name: {}'.format(current_file))
                else:
                    logger.info('Unused organ name: {}'.format(current_file))

            if 'eye' in current_file.lower():
                if current_file.lower().startswith('l') or current_file.endswith('lt') or current_file.endswith('left'):
                    # Lt_Eye.append(current_file)
                    organ_name_dict['Lt_Eye'].append((current_file, current_file_path))
                elif current_file.lower().startswith('r') or current_file.endswith('rt') or current_file.endswith(
                        'right'):
                    # Rt_Eye.append(current_file)
                    organ_name_dict['Rt_Eye'].append((current_file, current_file_path))
                else:
                    logger.info('Unused organ name: {}'.format(current_file))

            if 'cochlea' in current_file.lower():
                if current_file.lower().startswith('l') or current_file.endswith('lt') or current_file.endswith('left'):
                    # Lt_Cochlea.append(current_file)
                    organ_name_dict['Lt_Cochlea'].append((current_file, current_file_path))
                elif current_file.lower().startswith('r') or current_file.endswith('rt') or current_file.endswith(
                        'right'):
                    # Rt_Cochlea.append(current_file)
                    organ_name_dict['Rt_Cochlea'].append((current_file, current_file_path))
                else:
                    logger.info('Unused organ name: {}'.format(current_file))

            if 'opt' in current_file.lower() and 'nerve' in current_file.lower():
                if current_file.lower().startswith('l') or current_file.endswith('lt') or current_file.endswith('left'):
                    # Lt_OpticNerve.append(current_file)
                    organ_name_dict['Lt_OpticalNerve'].append((current_file, current_file_path))
                elif current_file.lower().startswith('r') or current_file.endswith('rt') or current_file.endswith(
                        'right'):
                    # Rt_OpticNerve.append(current_file)
                    organ_name_dict['Rt_OpticalNerve'].append((current_file, current_file_path))
                else:
                    logger.info('Unused organ name: {}'.format(current_file))

            if 'opt' in current_file.lower() and 'chiasm' in current_file.lower():
                # OpticChiasm.append(current_file)
                organ_name_dict['OpticalChiasma'].append((current_file, current_file_path))

            if 'oral' in current_file.lower() and 'cavity' in current_file.lower():
                # OralCavity.append(current_file)
                organ_name_dict['OralCavity'].append((current_file, current_file_path))

    return organ_name_dict


def organ_name_standardize(organ_name_dict, output_path, dataset_name):
    for current_organ_name, current_file_lists in organ_name_dict.items():
        for _, current_old_organ_path in current_file_lists:
            split_path = current_old_organ_path.split('\\')
            study_ID = split_path[-2]
            origin_organ_name = os.path.splitext(split_path[-1])[0]

            new_study_folder = os.path.join(output_path, study_ID)
            if not os.path.exists(new_study_folder):
                os.mkdir(new_study_folder)

            logger.info(f'Organ name standardization: {current_organ_name}, {study_ID}, {origin_organ_name}')
            if True:
                old_CT_path = '\\'.join(split_path[:-1] + ['CT.nrrd'])
                if not os.path.exists(os.path.join(new_study_folder, 'CT.nrrd')):
                    CT_data, old_CT_header = nrrd.read(old_CT_path)
                    old_CT_header['dataset'] = dataset_name
                    old_CT_header['study_ID'] = study_ID
                    old_CT_header['name'] = 'CT'
                    nrrd.write(os.path.join(new_study_folder, 'CT.nrrd'), CT_data, old_CT_header)

            mask_data, old_mask_header = nrrd.read(current_old_organ_path)
            old_mask_header['dataset'] = dataset_name
            old_mask_header['study_ID'] = study_ID
            old_mask_header['origin_organ_name'] = origin_organ_name
            old_mask_header['name'] = current_organ_name
            nrrd.write(os.path.join(new_study_folder, f'{current_organ_name}.nrrd'), mask_data, old_mask_header)


def get_bbox_from_mask(mask, target_size):
    mask = np.sum(mask, axis=0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    r_center, c_center = (rmin + rmax) // 2, (cmin + cmax) // 2
    rmin, rmax = r_center - target_size // 2, r_center + target_size // 2
    cmin, cmax = c_center - target_size // 2, c_center + target_size // 2

    rmin = np.clip(rmin, 0, mask.shape[0])
    rmax = np.clip(rmax, 0, mask.shape[0])

    cmin = np.clip(cmin, 0, mask.shape[1])
    cmax = np.clip(cmax, 0, mask.shape[1])
    return (rmin, rmax, cmin, cmax)


def get_directions(volume_data, slice_number=50, threshold=300):
    direction = (np.sum(volume_data[-slice_number:] > threshold) - np.sum(volume_data[:slice_number] > threshold)) > 0
    return direction


def data_unification(data_path, output_folder, output_size, output_resolution):
    target_size = output_size[1]
    target_slice_number = output_size[0]

    tmp_target_size = 320
    for roots, dirs, files in os.walk(data_path):
        if len(files) == 0:
            continue
        study_ID = roots.split('\\')[-1]
        logger.info('Data Unification: {}'.format(study_ID))

        current_CT_path = os.path.join(roots, 'CT.nrrd')
        current_CT, current_CT_header = nrrd.read(current_CT_path)
        current_spacing = current_CT_header['spacing']
        current_spacing = [float(x) for x in current_spacing[1:-1].split(',')]
        current_slice_thickness = float(current_CT_header['slice_thickness'])

        zoom_ratio_thickness = current_slice_thickness / output_resolution[0]
        if current_spacing[0] != current_spacing[1]:
            logger.info(f'Data Unification: spacing not equal! for {study_ID}')
        zoom_ratio = (zoom_ratio_thickness,
                      current_spacing[0] / output_resolution[1],
                      current_spacing[1] / output_resolution[2])

        # get directions
        if True:
            # todo: should get the directions from dicom files
            current_direction = get_directions(current_CT)
            if not current_direction:
                current_CT = current_CT[::-1]

        # get bbox
        mask_volume = 0
        mask_dict = {}
        for current_file in files:
            if current_file == 'CT.nrrd':
                continue
            current_mask, current_mask_header = nrrd.read(os.path.join(roots, current_file))
            mask_dict[current_file] = [current_mask, current_mask_header]
            mask_volume += current_mask
        rmin, rmax, cmin, cmax = get_bbox_from_mask(mask_volume, target_size=tmp_target_size)

        # todo: In theory, one should first zoom-in/zoom-out, then do crop. But this will increase the computation burden.
        current_CT = current_CT[:target_slice_number, rmin:rmax, cmin:cmax]
        current_CT = zoom(current_CT, zoom_ratio, order=1)
        current_CT = current_CT[:target_slice_number,
                     tmp_target_size // 2 - target_size // 2:tmp_target_size // 2 + target_size // 2,
                     tmp_target_size // 2 - target_size // 2:tmp_target_size // 2 + target_size // 2]
        current_CT_header['sizes'] = current_CT.shape
        current_CT_header['spacing'] = (1.0, 1.0)
        current_CT_header['slice_thickness'] = 3.0

        for k, v in mask_dict.items():
            if True:
                if not current_direction:
                    v[0] = v[0][::-1]
            v[0] = v[0][:target_slice_number, rmin:rmax, cmin:cmax]
            v[0] = zoom(v[0], zoom_ratio, order=1)
            v[0][v[0] > 0.5] = 1
            v[0] = v[0][:target_slice_number,
                   tmp_target_size // 2 - target_size // 2:tmp_target_size // 2 + target_size // 2,
                   tmp_target_size // 2 - target_size // 2:tmp_target_size // 2 + target_size // 2]

            v[1]['sizes'] = current_CT.shape
            v[1]['spacing'] = (1.0, 1.0)
            v[1]['slice_thickness'] = 3.0

        current_output_folder = os.path.join(output_folder, study_ID)
        if not os.path.exists(current_output_folder):
            os.mkdir(current_output_folder)

        nrrd.write(os.path.join(current_output_folder, 'CT.nrrd'), current_CT, current_CT_header)
        for k, v in mask_dict.items():
            nrrd.write(os.path.join(current_output_folder, k), v[0], v[1])


def data_visualization(data_path, output_folder, vmin, vmax, is_1k_shift=True):
    for roots, dirs, files in os.walk(data_path):
        if len(files) == 0:
            continue
        study_ID = roots.split('\\')[-1]
        current_CT_path = os.path.join(data_path, study_ID, 'CT.nrrd')
        current_data, current_header = nrrd.read(current_CT_path)
        logger.info('Data visualization: {}, {}, {}, {}'.format(study_ID,
                                                                current_header['spacing'],
                                                                current_header['slice_thickness'],
                                                                current_header['sizes']))

        for current_file in files:
            if 'CT.nrrd' in current_file:
                continue

            current_organ = os.path.splitext(current_file)[0]
            current_mask, _ = nrrd.read(os.path.join(roots, current_file))
            for current_reduce, current_view in zip([(-1, 0), (-1, -1), (0, 0)], ['coronal', 'axial', 'sagittal']):
                max_index = np.argmax(np.sum(np.sum(current_mask, axis=current_reduce[0]), axis=current_reduce[1]))
                if current_view == 'coronal':
                    CT_data = current_data[:, max_index]
                    mask_data = current_mask[:, max_index]
                elif current_view == 'axial':
                    CT_data = current_data[max_index]
                    mask_data = current_mask[max_index]
                else:
                    CT_data = current_data[:, :, max_index]
                    mask_data = current_mask[:, :, max_index]

                RGB_data = convert_to_RGB255(CT_data, vmin=vmin, vmax=vmax, is_1k_shift=is_1k_shift)
                Colorized_data = apply_mask(RGB_data, mask_data, color=[1, 0, 0], alpha=0.5)

                plt.imshow(Colorized_data)
                plt.axis('off')
                if not os.path.exists(os.path.join(output_folder, current_organ, current_view)):
                    os.makedirs(os.path.join(output_folder, current_organ, current_view))

                plt.savefig(
                    os.path.join(output_folder, current_organ, current_view, f'{study_ID}.png'),
                    bbox_inches='tight')
                plt.close()


if __name__ == '__main__':
    dataset_name = 'HNSCC'
    target_size = (128, 256, 256)
    target_resolution = (3.0, 1.0, 1.0)

    data_path = r'./example_data'
    output_folder = r'./cleaned_data'
    visualization_folder = r'./visualization'

    tmp_data_path_1 = 'tmp1'
    tmp_data_path_2 = 'tmp2'

    if True:
        # step 0: provide the data list (one should write his/her own script to extract the data list)
        # default input: a folder contains many sub-folders, each sub-folder contains 1) CT folder; 2) RTStruct file
        data_list = extract_data_list(data_path)

        # step 1: extract CT and contours from dicom information
        # we will temporally create a folder to save CT/contours extracted from dicom. Here, the name of the organs are not
        # standardized, one may need manually create a name mapping based on some predefined rules
        logger.info('Extracting CT and contours from DICOM files ...')

        if os.path.exists(tmp_data_path_1):
            os.remove(tmp_data_path_1)
        os.mkdir(tmp_data_path_1)

        for current_study, current_study_info in data_list.items():
            logger.info('Extracting CT and contours from DICOM files: {}'.format(current_study))
            current_study_data = extract_CT(current_study_info['CT'])
            current_study_data = extract_Contours(current_study_info['RTSTRUCT'], current_study_data)
            save_tmp_data(current_study_data, os.path.join(tmp_data_path_1, current_study))

        # step 2: adjust the directions (superior-inferior, right-left, anterior-posterior)
        # can be read this information from the dicom and merge with step 1

        # step 3: standardize the organ names
        # need to predefine a name map
        if os.path.exists(tmp_data_path_2):
            os.remove(tmp_data_path_2)
        os.mkdir(tmp_data_path_2)

        organ_name_dict = organ_name_rules(tmp_data_path_1)
        organ_name_standardize(organ_name_dict, output_path=tmp_data_path_2, dataset_name=dataset_name)

    # step 4: unify the resolution/slice number
    # need to predefine the normal resolution and the normal number of slice
    # HN: resolution --- 1 * 1 * 3mm, size --- 256 * 256 * 128
    # prostate: resolution --- 1 * 1 * 3mm, size --- 512 * 512 * 128
    if os.path.exists(output_folder):
        os.remove(output_folder)
    os.mkdir(output_folder)
    data_unification(data_path=tmp_data_path_2,
                     output_folder=output_folder,
                     output_size=target_size,
                     output_resolution=target_resolution)

    # step 5: check the correctness by visualizing each organ in three views
    if os.path.exists(visualization_folder):
        os.remove(visualization_folder)
    os.mkdir(visualization_folder)
    data_visualization(data_path=output_folder,
                       output_folder=visualization_folder,
                       vmin=0,
                       vmax=80,
                       is_1k_shift=True)

    logger.info('Congrats! May the force be with you ...')
