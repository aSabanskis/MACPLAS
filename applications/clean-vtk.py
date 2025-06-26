#!/usr/bin/env pvpython

from paraview.simple import *
import os.path
from os import remove
from glob import glob


def get_files():
    vtk_files = glob("./result-*.vtk")
    return vtk_files


def post(case, delete):
    # read data
    filename = case
    f_new = filename.replace("result-", "cleaned-")

    print(filename, f_new)

    if os.path.exists(f_new):
        print("Skipping")
        if delete:
            remove(filename)
        return

    reader = LegacyVTKReader(FileNames=[filename])
    clean = CleantoGrid(Input=reader)

    SaveData(f_new, proxy=clean, ChooseArraysToWrite=0, FileType="Binary")

    if delete:
        remove(filename)

    Delete(clean)
    Delete(reader)


delete = 1  # 0

print("Starting vtk file conversion")
if delete:
    print("Deleting original files")

cases = get_files()
for case in cases:
    post(case, delete)

print("Done!")
