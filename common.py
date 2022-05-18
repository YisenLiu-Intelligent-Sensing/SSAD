"""
 @file   common.py
 @brief  Commonly used script
 @author Yisen Liu
Copyright (C) 2022 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""

########################################################################
# import python-library
########################################################################
import glob
import os

import yaml

########################################################################


# load parameter.yaml
########################################################################
def yaml_load():
    with open("parameter.yaml") as stream:
        param = yaml.safe_load(stream)
    return param
#########################################################################


# load dataset
def select_dirs(param):
    """
    param : dict
        parameter.yaml data

    """
    dir_path = os.path.abspath("{base}/*".format(base=param["data_directory"]))
    dirs = sorted(glob.glob(dir_path))

    return dirs
#########################################################################

