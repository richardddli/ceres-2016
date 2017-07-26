"""
This script finds matching images across different camera bands within a given 
time interval, merges their camera logs, and aligns the images. This method of 
stacking images is agnostic to the number of camera bands, which has allowed it 
to be applied to all of the company’s current (and future) camera systems.
"""

import os
import shutil
import glob
import time

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from improc import preprocess
from improc.gen import camsettings, dirfuncs
from improc.gis import imselect, imu
from improc.dbops import loader
from improc.imops import imio


def watch_transfer_status(flight_id, input_stage="selected", **kwargs):
    """
    Checks if all raw files have transferred over for a flight, and if so,
    begins stacking the images for image-level alignment.

    Parameters
    ----------
    flight_id : int
        Flight ID.
    input_stage : str
        Stage of images to be stacked and mosaicked. Valid options:
        raw, selected, or corrected
    """

    if not is_ila_system(flight_id):
        print("No ILA parameters for system flown in flight %i." % flight_id)
        return
    try:
        stack_cams = kwargs["stack_cams"]
        kwargs.pop("stack_cams")
    except KeyError:
        camera_df = loader.get_table("Cameras")
        sys_type = camera_df.loc[flight_id, "VNIR_cam"].lower()
        sys_serial = camera_df.loc[flight_id, "VNIR_serial"]
        stack_cams = camsettings.get_setting(sys_type, key1="ILA_order",
                                             serial=sys_serial)

    base_dir = dirfuncs.guess_base_dir(flight_id)
    dirfuncs.gen_data_dirs(flight_id, base_dir, camera_list=[sys_type])
    data_dirs = dirfuncs.gen_data_dirnames(flight_id, camera_list=stack_cams)

    while True:
        num_files = {}
        for cam in stack_cams:
            num_files[cam] = len(glob.glob(os.path.join(data_dirs[cam][input_stage],
                    "*" + camsettings.get_setting(cam, "specs", "ext"))))
        # check if ~equal number of images from all camera bands have transferred
	if min(num_files.values()) > 50 and \
           max(num_files.values()) - min(num_files.values()) < 10:
            gen_stacked_logs(flight_id)
            stack(flight_id, input_stage, **kwargs)
            preprocess.separate_by_shape(flight_id, sys_type)
            break
        else:
            print("# images transferred: " +
                  ", ".join(["%i (%s)" % (num_files[cam], cam) for cam in num_files]) +
                  ". Sleeping")
        for i in range(300):
            time.sleep(1)


def gen_stacked_logs(flight_id, stack_cams=None, ref_cam="ids_red",
                     gap_thres=3):
    """
    Generates a log of stacked images (i.e. IDS2 or IDS5) analogous to the
    imu files for single bands in the Pilot directories. Matches individual
    images across bands using the given time-gap threshold. Retains full
    log information of the reference band, and appends imageNames, dateTime,
    and offset information from all other bands.

    Parameters
    ----------
    flight_id : int
        Flight ID
    stack_cams : list
        The bands to be stacked (in the given order). If None, uses
        the ILA order in camsettings.
    ref_cam : str
        Reference band to which all other bands were aligned in selecting
        the GCPs in camsettings. By default, the red band.
    gap_thres : int
        The maximum time-gap (in millisec) allowed between images to
        qualify for alignment.

    Output
    ------
    None
    """

    if not is_ila_system(flight_id):
        print("No ILA parameters for system flown in flight %i." % flight_id)
        return

    camera_df = loader.get_table("Cameras")
    sys_type = camera_df.loc[flight_id, "VNIR_cam"].lower()
    sys_serial = camera_df.loc[flight_id, "VNIR_serial"]
    if stack_cams is None:
        stack_cams = camsettings.get_setting(sys_type, key1="ILA_order",
                                             serial=sys_serial)
    data_dirs = dirfuncs.gen_data_dirnames(flight_id, camera_list=stack_cams)
    logs = {cam: imselect.read_log(os.path.join(data_dirs["base"],
        camsettings.get_setting(cam, "logs", "imu"))) for cam in stack_cams}

    # generate array of matching image indices across all bands
    matched_ims = pd.DataFrame(index=range(len(logs[ref_cam])))
    for cam in stack_cams:
        pairs = find_pairs(logs[ref_cam]["dateTime"].values,
                           logs[cam]["dateTime"].values, gap_thres)
        matched_ims = pd.concat([matched_ims, pairs.rename(cam)], axis=1)
    # remove unmatched images
    matched_ims = matched_ims.dropna().astype(int)
    print("Images matched: %i/%i" % (len(matched_ims),
                                     min([len(log) for log in logs.values()])))

    stacked_imu = logs[ref_cam].iloc[matched_ims.index].reset_index()
    stacked_imu['imageNames'] = stacked_imu['imageNames'].apply(lambda f:
        f.replace(camsettings.get_setting(ref_cam, "specs", "ext"),
                  camsettings.get_setting(sys_type, "specs", "ext")))

    for cam in stack_cams:
        cam_imu = logs[cam].iloc[matched_ims[cam]].reset_index()
        cam_imu = cam_imu.loc[:, ["imageNames", "dateTime"]]
        cam_imu["offset(ms)"] = ((stacked_imu["dateTime"] -
                                  cam_imu["dateTime"]) /
                                 np.timedelta64(1, "ms"))
        new_columns = {s: "%s_%s" % (cam[-3:], s) for s in cam_imu.keys()}
        cam_imu.rename(columns=new_columns, inplace=True)
        stacked_imu = pd.concat([stacked_imu, cam_imu], axis=1)
    stacked_imu.drop(["index", "geometry"], axis=1, inplace=True)

    out_file = os.path.join(data_dirs["base"],
        camsettings.get_setting(sys_type, "logs", "imu", serial=sys_serial))
    stacked_imu.to_csv(out_file, index=False)
    print("Generated log file: %s" % out_file)


def stack(flight_id, input_stage, stack_cams=None, ref_cam="ids_red",
          gap_thres=3, trim_size=None, skip_existing=False, **kwargs):
    """
    Performs image-level alignment of specified bands for all images
    within a given time-gap threshold.

    Parameters
    ----------
    flight_id : int
        Flight ID.
    input_stage : str
        Stage of images to be stacked and mosaicked. Valid options:
        raw, selected, or corrected
    stack_cams : list
        The bands to be stacked (in the given order). If None, uses
        the ILA order in camsettings.
    ref_cam : str
        Reference band to which all other bands were aligned in selecting
        the GCPs in camsettings. By default, the red band.
    gap_thres : int
        The maximum time-gap (in millisec) allowed between images to
        qualify for alignment.
    trim_size : int
        The number of pixels to trim off each edge after alignment, to
        remove black space from the alignment. By default, 75 for
        selected (untrimmed) images and 25 for corrected (trimmed) or
        pomona images.
    skip_existing : bool
        If True, does not overwrite any existing stacked or trimmed images.

    Returns
    -------
    None
    """

    if not is_ila_system(flight_id):
        print("No ILA parameters for system flown in flight %i." % flight_id)
        return

    camera_df = loader.get_table("Cameras")
    sys_type = camera_df.loc[flight_id, "VNIR_cam"].lower()
    sys_serial = camera_df.loc[flight_id, "VNIR_serial"]
    if stack_cams is None:
        stack_cams = camsettings.get_setting(sys_type, "ILA_order",
                                             serial=sys_serial)

    base_dir = dirfuncs.guess_base_dir(flight_id)
    dirfuncs.gen_data_dirs(flight_id, base_dir, camera_list=[sys_type])
    dirfuncs.gen_processing_dirs(flight_id)
    cl = stack_cams + [sys_type]
    data_dirs = dirfuncs.gen_data_dirnames(flight_id, camera_list=cl)

    stacked_imu_file = os.path.join(data_dirs["base"],
                camsettings.get_setting(sys_type, "logs", "imu", sys_serial))
    if not os.path.exists(stacked_imu_file):
        print("No log file found. Generating %s" % stacked_imu_file)
        gen_stacked_logs(flight_id, stack_cams=stack_cams, ref_cam=ref_cam,
                         gap_thres=gap_thres)
    a, b, c, d = preprocess.select_by_shape(flight_id, sys_type)
    sel_imu = pd.concat(d)

    homographies = {}
    for cam in stack_cams:
        if not glob.glob(os.path.join(data_dirs[cam][input_stage],
                         "*" + camsettings.get_setting(cam, "specs", "ext"))):
            print("No %s images found for %s band." % (input_stage, cam[-3:]))
            return
        if cam != ref_cam:
            base_points, warp_points = \
                [np.array(pts) for pts in camsettings.get_setting(sys_type,
                             key1="warps", key2=cam[-3:], serial=sys_serial)]
            if input_stage == "corrected":
                base_points -= 75
                warp_points -= 75
            homographies[cam] = cv2.findHomography(
                                    base_points, warp_points, method=0,
                                    ransacReprojThreshold=3.0)[0]

    cor_dir = data_dirs[sys_type]["corrected"]
    ref_ext = camsettings.get_setting(sys_type, "specs", "ext", sys_serial)

    for index, row in sel_imu.iterrows():
        im_out = None
        filename_out = os.path.join(cor_dir, row["imageNames"])
        if skip_existing and os.path.exists(filename_out):
            continue
        corrupt_image = False
        for cam_index, cam in enumerate(stack_cams):
            try:
                band_file = os.path.join(data_dirs[cam][input_stage],
                                         row[cam[-3:] + "_imageNames"])
                band = imio.imread(band_file)
                if im_out is None:
                    width, height = band.shape[0:2]
                    im_out = np.zeros((width, height, len(stack_cams)),
                                      dtype="uint16")
            except OSError:
                print("Skipping corrupt or missing image: %s" % band_file)
                imu.mark_corrupt_image(flight_id, band_file, cam, 
                                       stacked_sys=sys_type)
                corrupt_image = True
                break
            if cam == ref_cam:
                im_out[:, :, cam_index] = band
            else:
                band_warped = cv2.warpPerspective(band, homographies[cam],
                                                  (height, width))
                im_out[:, :, cam_index] = band_warped
        if corrupt_image:
            continue
        if trim_size is None:
            if input_stage == 'selected':
                trim_size = 75
            else:
                trim_size = camsettings.get_setting(sys_type, "warps",
                                                    "trim_size",
                                                    serial=sys_serial)
        im_out = imio.trim(im_out, trim_size)
        imio.imsave(filename_out, im_out)

    print("\nSuccessfully stacked %i/%i images: %s"
          % (len(glob.glob(os.path.join(cor_dir, "*" + ref_ext))),
             len(sel_imu), cor_dir))

    ILA_dir = os.path.join(dirfuncs.get_processing_dirs(flight_id)["mosaic"],
                           "ILA")
    if not os.path.exists(ILA_dir):
        os.mkdir(ILA_dir)


def find_pairs(band1_times, band2_times, gap_thres):
    """
    Finds all pairs of images between two bands that fall within the gap
    threshold.

    Parameters
    ----------
    band1_times : list
        List of times for all images in band 1 (in ascending order)
    band2_times : list
        List of times for all images in band 2 (in ascending order)
    gap_thres : int
        Maximum threshold for time gap between pair of images (in millisec)

    Returns
    -------
    pairs : Series
        Pandas Series mapping the index of band1 file to that of band2.
    """

    pairs = pd.Series(index=range(len(band1_times)))
    index1, index2 = 0, 0

    while True:
        if (index1 == len(band1_times) or index2 == len(band2_times)):
            break
        diff = band1_times[index1] - band2_times[index2]
        if abs(diff) < np.timedelta64(gap_thres, "ms"):
            pairs.loc[index1] = index2
            index1 += 1
            index2 += 1
        elif diff < np.timedelta64(0, 'ns'):
            index1 += 1
        else:
            index2 += 1

    return pairs



def plot_time_offsets(flight_id, ref_cam="Red", cams=["NIR", "Red"],
                      gap_thres=13):
    ref_path = os.path.normpath(dirfuncs.get_data_dir(flight_id, 0, "ids_red"))
    paths, files, times = {}, {}, {}
    for cam in cams:
        paths[cam] = ref_path.replace(ref_cam, cam)
        imupath = os.path.join(os.path.dirname(os.path.dirname(paths[cam])),
                               "ids%sImuData.txt" % cam)
        files[cam] = glob.glob(os.path.join(paths[cam], "*.png"))
        files[cam].sort()
        times[cam] = time_from_filename(files[cam], imupath)

    matched_imgs = {ref_idx: [] for ref_idx in range(len(files[ref_cam]))}
    for cam in cams:
        pairs = find_pairs(times[ref_cam], times[cam], gap_thres)
        for ref_idx in list(matched_imgs):
            if ref_idx not in pairs:
                del matched_imgs[ref_idx]
            else:
                matched_imgs[ref_idx].append(pairs[ref_idx])

    for cam in cams:
        if cam is ref_cam:
            continue
        offsets = []
        for img in matched_imgs:
            offsets.append(times[ref_cam][img] -
                           times[cam][matched_imgs[img][cams.index(cam)]])
        plt.hist(offsets, bins=150, histtype="step", label=ref_cam + "-" + cam)
        plt.legend()



