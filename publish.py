""" 
Functions to publish processed images and deliver to customers via
Mapbox API.
"""

import glob
import json
import os
import shutil

import pandas as pd
import numpy as np

from improc.dbops import access, finder, loader, parse, spatial, edit
from improc.dbops import utils as dbutils
from improc.gen import dirfuncs, strops, transfer, camsettings, fileio
from improc.web import credentials, mapbox, sendmail
from improc.gis import coords
from improc import postprocess
from improc import rastertools


def prepare_field(flight_id, field, replace=True):
    """
    Generates WebApp directory for field, and copies all available files
    in color merged to the WebApp directory before publication.

    Parameters
    ----------
    flight_id : int

    field : int or str

    Returns
    -------
    files : list
        New filenames in WebApp directory.
    """

    field_id = dbutils.idify(field, "Fields")
    merge_dir = dirfuncs.guess_flight_dir(flight_id, "color merged")
    if merge_dir is None:
        raise ValueError("Color merged directory does not exist " +
                         "for flight %i." % flight_id)
    merge_files = (glob.glob(merge_dir + "*.tif") +
                   glob.glob(merge_dir + "*/*.tif"))
    field_merge_files = [merge_file for merge_file in merge_files if
                         parse.get_fid_from_filename(merge_file) == field_id]
    field_webapp_dir = gen_webapp_dir(flight_id, field)

    webapp_files = [field_webapp_dir + os.path.basename(f)
                    for f in field_merge_files]

    for (merge_file, webapp_file) in zip(field_merge_files, webapp_files):
        if os.path.exists(webapp_file) and not replace:
            continue
        shutil.copy(merge_file, webapp_file)

    return webapp_files


def prepare_flight(flight_id):
    """
    Generates WebApp directory for field, and copies all available files
    in color merged to the WebApp directory before publication.

    Parameters
    ----------
    flight_id : int

    Returns
    -------
    files : list
        New filenames in WebApp directory.
    """

    field_ids = access.get_fields_flown(flight_id)
    webapp_files = []

    for fid in field_ids.values:
        field_files = prepare_field(flight_id, fid)
        webapp_files.append(field_files)

    return webapp_files


“””
The field, user, and date JSON files are required for
proper display of each layer on the webapp.
“””

def gen_field_json(field, webapp_id=0, replace=True, permissions=None):
    """
    Generates field.json file for a particular field, using database
    information.

    Parameters
    ----------
    field : int or str

    webapp_id : int
        Usually 0.
    replace : bool
        Replace the existing field.json file.
    permission : str
        Provide access to this field from child accounts.

    Returns
    -------
    field_json : str
        Full filepath to field.json file.
    """

    field_id = dbutils.idify(field, "Fields")
    fields = loader.get_table("Fields")
    customers = loader.get_table("Customers")

    field_name = fields.loc[field_id]["Field_Name"]
    cust_id = fields.loc[field_id]["Cust_ID"]
    cust_name = customers.loc[cust_id]["Cust_Name"]

    field_dir = os.path.join(dirfuncs.guess_web_app_dir(), cust_name,
                             field_name)
    field_json = os.path.join(field_dir, "field.json")
    field_lon, field_lat = spatial.get_center_coords(field_id)

    if os.path.exists(field_json):
        data = json.load(open(field_json))
    else:
        data = {}
    if replace:
        data["name"] = field_name
        data["Field_ID"] = webapp_id
        data["zoom"] = determine_zoom(field_id)
        data["lat"] = field_lat
        data["lng"] = field_lon
        if permissions is not None:
            data["permissions"] = [permissions]
        with open(field_json, mode='w') as out_file:
            json.dump(data, out_file)

    return field_json


def gen_user_json(cust):
    """
    Generates user.json file in the customer directory in web_app, using
    database information. If no email provided in Customers.csv, generates
    generic gmail address.

    Parameters
    ----------
    cust : int or str

    Returns
    -------
    user_json : str
        Full filepath to user.json file
    """

    cust_id = dbutils.idify(cust, "Customers")
    customers = loader.get_table("Customers")
    cust_name = customers.loc[cust_id]["Cust_Name"]
    cust_email = customers.loc[cust_id]["Email"]

    if pd.isnull(cust_email):
        cust_email = "".join(cust_name.lower().split()) + "@gmail.com"

    cust_dir = os.path.join(dirfuncs.guess_web_app_dir(), cust_name)
    user_json = os.path.join(cust_dir, "user.json")

    data = {}
    data["name"] = cust_name
    data["id"] = cust_email

    with open(user_json, mode='w') as out_file:
        json.dump(data, out_file)

    return user_json


def get_date_json_file(flight_id, field):
    """
    Creates directories in web_app folder as necessary and returns full path of
    date.json file for one field.

    Parameters
    ----------
    flight_id : int

    field : int or str

    Returns
    -------
    date_json : str
        Full path to the date.json file for the given flight/field combo.
    """

    web_app_dir = dirfuncs.guess_web_app_dir()
    if web_app_dir is None:
        raise ValueError("web_app directory not found on this computer.")

    field_id = dbutils.idify(field, "Fields")
    fields = loader.get_table("Fields")
    flights = loader.get_table("Flights")
    customers = loader.get_table("Customers")

    # get required info about customer/flight from database
    customer_id = fields.loc[field_id, "Cust_ID"]
    customer_name = customers.loc[customer_id, "Cust_Name"]
    flight_date = flights.loc[flight_id, "Date"]
    field_name = fields.loc[field_id, "Field_Name"]

    # create directories, if necessary
    customer_dir = os.path.join(web_app_dir, customer_name)
    dirfuncs.gen_dir(customer_dir)
    field_dir = os.path.join(customer_dir, field_name)
    dirfuncs.gen_dir(field_dir)
    date_dir = os.path.join(field_dir, flight_date)
    dirfuncs.gen_dir(date_dir)

    user_json = os.path.join(customer_dir, "user.json")
    if not os.path.exists(user_json):
        gen_user_json(customer_id)

    field_json = os.path.join(field_dir, "field.json")
    if not os.path.exists(field_json):
        gen_field_json(field_id)

    date_json = os.path.join(date_dir, "date.json")

    return date_json


def update_date_json(flight_id, field, layer_type, date_json=None):
    """
    Create, or update, the date.json file for a flight/field combo with
    the metadata to show a specific layer.

    Parameters
    ----------
    flight_id : int
        Flight ID as in database. For dual system flights, either ID will work.
    field_id : int
        Field ID as in database.
    layer_type : str
        String specifying layer type for correct display in the webapp.
        `json_mappings` contains valid options, which include "ndvi", "flir"
        "nikon".
    date_json : str (opt)
        Full path to date.json file. Should not generally be necessary to set
        it, as it can be determined via database info.

    Returns
    -------
    None
    """

    json_mappings = {"ndvi": "NDVI",
                     "NDVI": "NDVI",
                     "flir": "Water Stress",
                     "classified": "Water Stress",
                     "WSC": "Water Stress",
                     "CIgC": "Chlorophyll Classification",
                     "CIreC": "Chlorophyll Classification",
                     "TR": "Thermal",
                     "nikon": "RGB",
                     "RGB": "RGB",
                     "CIR": "Color Infrared",
                     "CIgreen": "Chlorophyll Index",
                     "CIre": "Chlorophyll Index",
                     "chl5": "Chlorophyll Index",
                     "NC": "NDVI Classification"}

    field_id = dbutils.idify(field, "Fields")
    if date_json is None:
        date_json = get_date_json_file(flight_id, field)

    flight = loader.get_table("Flights")
    flight_date = flight.loc[flight_id, "Date"]

    url_base = mapbox.mapbox_url + str(field_id) + "-" + flight_date
    url_end = "/{z}/{x}/{y}.png"

    if os.path.exists(date_json):
        with open(date_json, "r") as date_file:
            data = json.load(date_file)
    else:
        data = {}

    data[json_mappings[layer_type]] = "".join([url_base, "-", layer_type,
                                               url_end])

    with open(date_json, "w") as date_file:
        json.dump(data, date_file)


“””
Special case handling for customers who require nonstandard publishing.
“””

def prepare_aws_upload(flight_id, cust, field_ids=None,
                       skip_existing=False, upload=False, **kwargs):
    """
    Moves unmasked registered images into AWS directory for upload, and
    renames files per Gallo's specifications. For thermal imagery, converts
    DN into physical. AWS uploads images to appropriate bucket.
    
    TODO: messy and needs to be refactored

    cust: int or str
        This function is currently only setup to run Gallo (80) or
        Paramount (37) fields.
    field_ids: list
        If None specified, will search flight for fields corresponding to
        customer.
    skip_existing: bool
        If True, will not overwrite any physical files or images already
        present in the AWS folder. Use False if running on a field that has
        already undergone normal processing - else the files in physical will
        be masked.
    upload: bool
        If True, will subsequently AWS upload files in AWS directory to
        appropriate bucket.
    """

    proc_dirs = dirfuncs.get_processing_dirs(flight_id)
    aws_dir = os.path.join(proc_dirs[''], 'AWS')
    if not os.path.exists(aws_dir):
        os.makedirs(aws_dir)

    cam_settings = camsettings.PARAMS
    camera_df = loader.get_table('Cameras')
    vnir_system = camera_df.loc[flight_id, "VNIR_cam"]
    tr_key = access.get_camera_list(flight_id, ['ir'])[0]
    tr_system = next(iter(cam_settings[tr_key].values()))['name']

    if type(cust) is int:
        customers = loader.get_table('Customers')
        cust = customers.loc[cust].Cust_Name
    # identify relevant fields in flight
    if field_ids is None:
        fields_flown = list(access.get_fields_flown(flight_id))
        field_ids = []
        for fid in fields_flown:
            if cust in access.get_customer_from_field(fid)[1]:
                field_ids.append(fid)

    vnir_files = []
    tr_files = []
    for fid in field_ids:
        vnir_files.extend(glob.glob(
            os.path.join(proc_dirs['registered'],
                         '*%i*%s.tif' % (fid, vnir_system))))
        tr_files.extend(glob.glob(
            os.path.join(proc_dirs['registered'],
                         '*%i*%s.tif' % (fid, tr_system))))

    # generate physical imagery
    if skip_existing:
        tr_files = [f for f in tr_files
                    if not os.path.exists(f.replace('registered', 'physical'))]
    phys_out = postprocess.gen_temperature_files(tr_files, in_dir='registered')

    print('\nCreated physical files:')
    [print(f) for f in phys_out]
    phys_files = []
    for fid in field_ids:
        phys_files.extend(glob.glob(
            os.path.join(proc_dirs['physical'],
                         '*%i*%s.tif' % (fid, tr_system))))
    ndvi_files = []

    aws_temp_dir = os.path.join(proc_dirs[''], 'AWS temp')
    if cust == 'Paramount':
        if not os.path.exists(aws_temp_dir):
            os.makedirs(aws_temp_dir)
        ndvi_files = postprocess.gen_ndvi_files(vnir_files,
                                in_dir='registered', out_dir='output')
        for f in vnir_files + phys_files + ndvi_files:
            shp = finder.get_field_shapefile(parse.get_fid_from_filename(f))[0]
            in_dir = dirfuncs.guess_processing_stage(flight_id, f)
            rastertools.split_mask_and_write(f, shp, buf_size=0.001,
                                             in_dir=in_dir, out_dir='AWS temp',
                                             out_key='subfield')
        vnir_files = glob.glob(os.path.join(aws_temp_dir, '*tif'))
        phys_files, ndvi_files = [], []

    suffix_mappings = {"Gallo": {tr_system: 'Thermal',
                                 vnir_system: vnir_system},
                       "Paramount": {tr_system: 'TR',
                                     vnir_system: 'NIR',
                                     'NDVI': 'NDVI'}}
    paramount_prefixes = {'134': 'CE_PFC-',
                          '421': 'CE_WV-',
                          '659': 'CE-'}

    edit.update_flight(flight_id)
    print('\nRenamed and copied files:')
    for f in vnir_files + phys_files + ndvi_files:
        f_split = os.path.basename(f).split(' ')
        field_id = parse.get_fid_from_filename(f)
        edit.update_field(fid)
        t_avg = access.get_visit_datetime(flight_id, field_id)[1]

        if cust == 'Paramount':
            f_split = f_split[:2] + f_split[-2:]
            f_split[-2] = paramount_prefixes[f_split[1]] + f_split[-2]
        if cust == 'Gallo':
            f_split[0] = f_split[0].replace('-', '_')
        f_split[1] = '%02i%02i' % (t_avg.hour, t_avg.minute)
        f_split[-1] = f_split[-1].replace(f_split[-1].split('.')[0],
                    suffix_mappings[cust][f_split[-1].split('.')[0]])
        f_out = os.path.join(aws_dir, '_'.join(f_split))

        if skip_existing and os.path.exists(f_out):
            continue
        shutil.copy(f, f_out)
        print(f_out)

    if os.path.exists(aws_temp_dir):
        shutil.rmtree(aws_temp_dir)

    if upload:
        processed_images(flight_id, cust, **kwargs)
