import rasterio
import numpy as np
from pathlib import Path
import geopandas as gpd
from rasterio.features import geometry_mask


def get_tif_path(folder_path, satellite_index=0):
    if satellite_index == 0:
        BAND_NAME = np.array(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    elif satellite_index == 1:
        BAND_NAME = np.array(["B2", "B3", "B4", "B8", "B11", "B12"])
    else:
        return [None]*6
    BAND_INDEX = {name: i for i, name in enumerate(BAND_NAME)}

    folder_path = Path(folder_path)
    image_paths = [None]*6
    tif_paths = [path for path in folder_path.iterdir() if path.is_file()
                 and path.suffix.lower() == '.tif']

    for path in tif_paths:
        band_name = path.stem.split('_')[-1]
        if satellite_index == 1:
            band_name = path.stem.split('_')[0]
            # print(band_name)
        if band_name in BAND_NAME:
            image_paths[BAND_INDEX[band_name]] = path
            # print(path)
    return image_paths


def load_image_by_xy(image_paths, x=None, y=None, width=None, height=None):
    datas = []
    print(image_paths)

    with rasterio.open(image_paths[0]) as src:
        if x is None or y is None or width is None or height is None:
            window = None
            transform = src.transform
            profile = src.profile.copy()
            shape = (src.height, src.width)
        else:
            window = rasterio.windows.Window(
                x, y, width, height
            )
            transform = rasterio.windows.transform(window, src.transform)
            profile = src.profile.copy()
            profile.update({
                'width': width,
                'height': height,
                'transform': transform
            })
            shape = (height, width)
        crs = src.crs

    for path in image_paths:
        with rasterio.open(path) as src:
            data = src.read(1, window=window)
            datas.append(data)

    datas = np.stack(datas, axis=0)
    return datas, profile, crs, shape


def load_image_by_lonlat(image_paths, center_lon=None, center_lat=None, width=None, height=None):
    """
    根据经纬度加载图像（适用于地理坐标系TIF）

    :param image_paths: 图像路径列表
    :param center_lon: 中心经度（度）
    :param center_lat: 中心纬度（度）
    :param width: 裁剪宽度（像素）
    :param height: 裁剪高度（像素）
    :return: datas, profile, crs, shape
    """
    datas = []

    with rasterio.open(image_paths[0]) as src:
        if not src.crs.is_geographic:
            raise ValueError(f"TIF不是地理坐标系，当前坐标系: {src.crs}")

        crs = src.crs
        transform = src.transform
        profile = src.profile.copy()

        if center_lon is None or center_lat is None or width is None or height is None:
            window = None
            shape = (src.height, src.width)
        else:
            row, col = src.index(center_lon, center_lat)

            col_start = max(0, col - width // 2)
            row_start = max(0, row - height // 2)
            col_end = min(src.width, col_start + width)
            row_end = min(src.height, row_start + height)

            if col_end - col_start < width:
                col_start = max(0, col_end - width)
            if row_end - row_start < height:
                row_start = max(0, row_end - height)

            window = rasterio.windows.Window(
                col_start, row_start, col_end - col_start, row_end - row_start)
            transform = rasterio.windows.transform(window, src.transform)
            profile.update(
                {'width': window.width, 'height': window.height, 'transform': transform})
            shape = (window.height, window.width)

    for path in image_paths:
        with rasterio.open(path) as src:
            datas.append(src.read(1, window=window))

    datas = np.stack(datas, axis=0)
    return datas, profile, crs, shape


def load_image_by_projection(image_paths, center_x=None, center_y=None, width=None, height=None):
    """
    根据投影坐标加载图像（适用于投影坐标系TIF）

    :param image_paths: 图像路径列表
    :param center_x: 中心点X坐标（米）
    :param center_y: 中心点Y坐标（米）
    :param width: 裁剪宽度（像素）
    :param height: 裁剪高度（像素）
    :return: datas, profile, crs, shape
    """
    datas = []

    with rasterio.open(image_paths[0]) as src:
        if src.crs.is_geographic:
            raise ValueError(f"TIF不是投影坐标系，当前坐标系: {src.crs}")

        crs = src.crs
        transform = src.transform
        profile = src.profile.copy()

        if center_x is None or center_y is None or width is None or height is None:
            window = None
            shape = (src.height, src.width)
        else:
            row, col = src.index(center_x, center_y)

            col_start = max(0, col - width // 2)
            row_start = max(0, row - height // 2)
            col_end = min(src.width, col_start + width)
            row_end = min(src.height, row_start + height)

            if col_end - col_start < width:
                col_start = max(0, col_end - width)
            if row_end - row_start < height:
                row_start = max(0, row_end - height)

            window = rasterio.windows.Window(
                col_start, row_start, col_end - col_start, row_end - row_start)
            transform = rasterio.windows.transform(window, src.transform)
            profile.update(
                {'width': window.width, 'height': window.height, 'transform': transform})
            shape = (window.height, window.width)

    for path in image_paths:
        with rasterio.open(path) as src:
            datas.append(src.read(1, window=window))

    datas = np.stack(datas, axis=0)
    return datas, profile, crs, shape


def save_image(output_paths, datas, meta):
    meta.update({
        'count': len(datas)
    })
    with rasterio.open(output_paths, 'w', **meta) as dst:
        dst.write(datas)


def get_shp_mask(shp_path, crs, shape, transform):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    mask = geometry_mask(geometries=gdf.geometry,
                         out_shape=shape, transform=transform, invert=True)
    return mask
