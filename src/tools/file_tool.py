from pathlib import Path
import shutil
import zipfile
import rasterio
from rasterio.enums import Resampling
import numpy as np
import time
from tqdm import tqdm


def find_zip_files(file_path):
    """
    查找路径下的所有zip格式文件 
    :param file_path:文件路径
    :return:包含压缩包路径的列表
    """
    file_path = Path(file_path)
    zip_files = list(file_path.glob("**/*.zip"))
    return zip_files


def decompress(file, temp_dir):
    """
    对 zip 格式的文件进行解压缩，并保存到临时目录中。

    :param file: zip 格式的压缩包路径（应为 Path 对象或字符串）
    :param temp_dir: 临时解压目录
    :return: 解压缩后生成的文件夹路径（Path 对象）
    """
    file_name = file.stem + '_unzip'
    un_zipfile = Path(temp_dir) / file_name
    # 判断文件夹是否存在，存在则先删除
    if un_zipfile.is_dir():
        shutil.rmtree(un_zipfile)

    with zipfile.ZipFile(file, mode='r') as zip_data:
        # 创建新的文件夹
        un_zipfile.mkdir(parents=True)
        zip_data.extractall(un_zipfile)
        return un_zipfile


def find_jp2_files(file):
    """
    查找 Sentinel-2 解压后路径下的所有 .jp2 文件，
    并按分辨率筛选出指定波段：
      - 10m: ['B02', 'B03', 'B04', 'B08']
      - 20m: ['B05', 'B06', 'B07', 'B8A']
      - 60m: ['B01', 'B09', 'B11', 'B12']

    :param file: 解压缩后的 Sentinel-2 产品文件夹路径（如 S2A_MSIL1C_XXXX.SAFE）
    :return: 三个列表 (bands_r10, bands_r20, bands_r60)，每个元素是 Path 对象
    """
    file_path = Path(file)
    jp2 = list(file_path.glob("**/*.jp2"))

    img_data_r10 = [
        jp2_file for jp2_file in jp2
        if len(jp2_file.parts) > 7 and jp2_file.parts[-3] == 'IMG_DATA' and jp2_file.parts[-2] == 'R10m'
    ]

    img_data_r20 = [
        jp2_file for jp2_file in jp2
        if len(jp2_file.parts) > 7 and jp2_file.parts[-3] == 'IMG_DATA' and jp2_file.parts[-2] == 'R20m'
    ]

    img_data_r60 = [
        jp2_file for jp2_file in jp2
        if len(jp2_file.parts) > 7 and jp2_file.parts[-3] == 'IMG_DATA' and jp2_file.parts[-2] == 'R60m'
    ]

    bands_r10 = [
        band for band in img_data_r10
        if band.stem.split('_')[-2] in ['B02', 'B03', 'B04', 'B08']
    ]

    bands_r20 = [
        band for band in img_data_r20
        if band.stem.split('_')[-2] in ['B05', 'B06', 'B07', 'B8A']
    ]

    bands_r60 = [
        band for band in img_data_r60
        if band.stem.split('_')[-2] in ['B01', 'B09', 'B11', 'B12']
    ]

    return bands_r10, bands_r20, bands_r60


def jp2_to_tif(bands_r10, bands_r20, bands_r60, output_dir, zip_name):
    """
    读取 JP2 格式的 Sentinel-2 波段数据：
      - 10m 波段直接使用
      - 20m 波段重采样到 10m（×2）
      - 60m 波段重采样到 10m（×6）
    然后按顺序合并为 12 波段 GeoTIFF，每个波段单独保存为 B1.tif ~ B12.tif

    波段顺序: ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']

    :param bands_r10: list of Path, ['B02','B03','B04','B08'] (10m)
    :param bands_r20: list of Path, ['B05','B06','B07','B8A'] (20m)
    :param bands_r60: list of Path, ['B01','B09','B11','B12'] (60m)
    :param output_dir: 输出目录
    :param zip_name: zip文件名（用于创建子文件夹）
    :return: 输出文件夹路径
    """
    band10 = []
    band20 = []
    band60 = []
    profile = None

    # === 读取 10m 波段（B02, B03, B04, B08）===
    for band in bands_r10:
        with rasterio.open(band) as src:
            data = src.read().astype('float32')  # shape: (1, H, W)
            if profile is None:
                profile = src.profile.copy()
            band10.append(data)

    # === 读取 20m 波段并重采样到 10m（×2）===
    for band in bands_r20:
        with rasterio.open(band) as src:
            # 目标尺寸：高度和宽度都 ×2
            out_shape = (src.count, src.height * 2, src.width * 2)
            data = src.read(
                out_shape=out_shape,
                resampling=Resampling.bilinear
            ).astype('float32')
            band20.append(data)

    # === 读取 60m 波段并重采样到 10m（×6）===
    for band in bands_r60:
        with rasterio.open(band) as src:
            # 目标尺寸：高度和宽度都 ×6
            out_shape = (src.count, src.height * 6, src.width * 6)
            data = src.read(
                out_shape=out_shape,
                resampling=Resampling.bilinear
            ).astype('float32')
            band60.append(data)

    # === 按指定顺序重组 12 个波段 ===
    # 输入顺序假设：
    #   band10 = [B02, B03, B04, B08]
    #   band20 = [B05, B06, B07, B8A]
    #   band60 = [B01, B09, B11, B12]
    #
    # 目标顺序: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
    bands = [
        band60[0],      # B01
        band10[0],      # B02
        band10[1],      # B03
        band10[2],      # B04
        band20[0],      # B05
        band20[1],      # B06
        band20[2],      # B07
        band10[3],      # B08
        band20[3],      # B8A
        band60[1],      # B09
        band60[2],      # B11
        band60[3]       # B12
    ]

    # === 创建输出文件夹 ===
    # 去掉zip扩展名，作为文件夹名
    folder_name = Path(zip_name).stem
    output_folder = Path(output_dir) / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # === 更新元数据 ===
    meta = profile.copy()
    meta.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1  # 每个文件单波段
    })

    # === 逐个保存为单波段 GeoTIFF ===
    for i, arr in enumerate(bands, start=1):
        # 波段名称：B1.tif, B2.tif, ..., B12.tif
        band_name = f"B{i}.tif"
        out_tif_path = output_folder / band_name

        # 处理B8A特殊命名（虽然保存为B9，但内容对应B8A）
        # 这里保持按顺序命名即可

        with rasterio.open(out_tif_path, 'w', **meta) as dst:
            dst.write(np.squeeze(arr), 1)

    return output_folder


def clean_temp_files(temp_dir):
    """
    清理临时解压目录

    :param temp_dir: 临时目录路径
    :return: None
    """
    temp_path = Path(temp_dir)
    if temp_path.is_dir():
        shutil.rmtree(temp_path)
        print(f"已清理临时目录: {temp_dir}")


def data_unzip(file_path, output_dir, keep_temp=False):
    """
    批量处理 Sentinel-2 压缩包：
      1. 查找所有 .zip 文件
      2. 逐个解压到临时目录
      3. 提取 B01-B12 共 12 个波段（按分辨率分类）
      4. 重采样到 10m 并保存为单波段 GeoTIFF（B1.tif ~ B12.tif）
      5. 删除解压后的临时文件（可选）

    波段顺序: ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
    输出文件: B1.tif ~ B12.tif

    :param file_path: 存放 Sentinel-2 .zip 压缩包的目录路径
    :param output_dir: 输出目录路径
    :param keep_temp: 是否保留临时解压文件，默认 False（删除）
    :return: None
    """
    start_time = time.time()

    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建临时目录
    temp_dir = Path(output_path) / "temp_unzip"
    temp_dir.mkdir(parents=True, exist_ok=True)

    zip_files = find_zip_files(file_path)
    print(f"找到 {len(zip_files)} 个待处理压缩包")
    print(f"输出目录: {output_path.absolute()}")

    for file in tqdm(zip_files, desc='处理进度', unit='文件'):
        try:
            # 解压到临时目录
            un_zipfile = decompress(file, temp_dir)

            # 查找JP2文件
            bands_r10, bands_r20, bands_r60 = find_jp2_files(un_zipfile)

            # 转换为TIF并保存到输出目录
            output_folder = jp2_to_tif(
                bands_r10, bands_r20, bands_r60, output_dir, file.name)

            # print(f"\n✓ {file.name} 已处理完成，输出到: {output_folder}")

            # 清理临时解压文件
            if not keep_temp:
                shutil.rmtree(un_zipfile)

        except Exception as e:
            print(f"\n✗ 处理 {file.name} 时出错: {str(e)}")
            continue

    # 清理临时目录
    if not keep_temp:
        clean_temp_files(temp_dir)
    else:
        print(f"临时文件保留在: {temp_dir}")

    end_time = time.time()
    total_time = end_time - start_time
    print('=' * 50)
    print('任务完成！')
    print(f"程序运行时间为：{total_time:.2f} 秒")
    print('=' * 50)
