from osgeo import gdal
import numpy as np

list1 = ["byte", "uint8", "uint16", "int16", "uint32", "int32",
         "float32", "float64", "cint16", "cint32", "cfloat32", "cfloat64"]
list2 = [gdal.GDT_Byte, gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
         gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]


def imgread(path) -> np.ndarray:
    img = gdal.Open(path)
    c = img.RasterCount
    img_arr = img.ReadAsArray()
    if c > 1:
        img_arr = img_arr.swapaxes(1, 0)
        img_arr = img_arr.swapaxes(2, 1)
    del img
    return img_arr


def imgwrite(path, narray, compress="None") -> None:
    s = narray.shape
    dt_name = narray.dtype.name
    # TODO: controllare questo for che fa una cosa strana
    for i in range(len(list1)):
        if list1[i] in dt_name.lower():
            datatype = list2[i]
            break
        else:
            datatype = list2[0]
    if len(s) == 2:
        row, col, c = s[0], s[1], 1
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(path, col, row, c, datatype, options=[
                                "COMPRESS="+compress])
        dataset.GetRasterBand(1).WriteArray(narray)
        del dataset
    elif len(s) == 3:
        row, col, c = s[0], s[1], s[2]
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(path, col, row, c, datatype)
        for i in range(c):
            dataset.GetRasterBand(i + 1).WriteArray(narray[:, :, i])
        del dataset
