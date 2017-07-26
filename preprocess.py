def select_by_image_attribute(flight_id, field_id, camera, image_attr):
    """
    Select images from the separated directory by one of its attributes
    in the log file: i.e. Altitude_asl(m), dateTime, bearing, etc. Allows
    for filtering out images that may create artifacts in processing
    (oblique images taken during turns, shadows too late in the day).
    “””

    Parameters
    ----------
    flight_id : int

    field_id : int

    camera : str

    image_attr: dict
        This dict contains the name of the attribute, and lower and upper
        bounds. The proper format is as follows:
        {'attr': 'Altitude_asl(m)', 'min': 800, 'max': 1200}

    Returns
    -------
    selected_tags : pandas DataFrame
        Images selected after applying selection criteria.

    Notes
    -----
    Overwrites imu file in separated directory with the selected images.
    Saves a copy of the original imu file appended with "Preselection".
    """

    imu_dir = dirfuncs.get_imu_dir(flight_id, field_id, camera)
    if not imu_dir:
        print('Imu directory does not exist')
        return
    imu_path = imu_dir + camsettings.get_setting(camera, key1="logs",
                                                 key2="imu")
    if os.path.exists(imu_path[:-4] + ' Preselection.txt'):
        image_tags = pd.read_csv(imu_path[:-4] + ' Preselection.txt', sep='\t')
    else:
        image_tags = pd.read_csv(imu_path, sep='\t')
        # backup old imu file in directory
        image_tags.to_csv(imu_path[:-4] + ' Preselection.txt',
                          index=False, sep='\t')

    expected_keys = ['attr', 'min', 'max']
    for key in expected_keys:
        if key not in image_attr:
            print('image_attr is missing key: %s' % key)
            return
    if image_attr['attr'] not in image_tags.keys():
        print(('%s is not a valid image attribute' % image_attr['attr']) +
              'in the imu file')
        return

    selected_tags = image_tags[(image_tags[image_attr['attr']] >
                                image_attr['min']) &
                               (image_tags[image_attr['attr']] <
                                image_attr['max'])]
    selected_tags.to_csv(imu_path, index=False, sep='\t')
    print('%i images found that match selection criteria. \nSaving %s'
          % (len(selected_tags.index), imu_path))

    return selected_tags