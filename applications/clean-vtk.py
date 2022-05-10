#!/usr/bin/env pvpython

from paraview.simple import *
import os.path
from glob import glob


def get_files():
    vtk_files = glob("./result-*.vtk")
    return vtk_files


def post(case):
    # read data
    filename = case
    f_new = filename.replace("result-", "cleaned-")

    print(filename, f_new)

    if not os.path.exists(filename):
        print("Skipping")
        return

    if os.path.exists(f_new):
        print("Skipping")
        return

    reader = LegacyVTKReader(FileNames=[filename])
    clean = CleantoGrid(Input=reader)

    SaveData(f_new, proxy=clean, ChooseArraysToWrite=0, FileType="Binary")

    Delete(clean)
    Delete(reader)


cases = get_files()
for case in cases:
    post(case)
