from pathlib import Path
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

src_root = Path(__file__).parent.parent
sys.path.append(str(src_root))

try:
    from tools import image_tool, calculate_tool
except ModuleNotFoundError as e:
    print(f"❌ 导入失败: {e}")

if __name__ == '__main__':
    folder_path = Path(r"D:\outputimg\2018")

    shp_path = Path(
        r"C:\Users\39632\Desktop\add_rockoutcrop_landsat_v7.3\add_rockoutcrop_landsat_v7.3.shp")
    paths = image_tool.get_tif_path(folder_path, satellite_index=1)

    # datas, profile, crs, shape = data.load_image_by_projection(paths, 529300, 2470080, 500, 500
    #                                                            )
    datas, profile, crs, shape = image_tool.load_image_by_projection(paths, 511720, 1715640, 500, 500
                                                                     )
    # mask = image_tool.get_shp_mask(
    #     shp_path, crs, shape, profile['transform'])

    # models_path = Path("models/v0.2")

    # rf = joblib.load(models_path / 'rfc_5.joblib')
    # xgbc = joblib.load(models_path / 'xgbc_5.joblib')

    blue = datas[0]/10000
    green = datas[1]/10000
    red = datas[2]/10000
    nir = datas[3]/10000
    swir1 = datas[4]/10000
    swir2 = datas[5]/10000

    cietx, ciety = calculate_tool.rgb2cie(red, green, blue)
    ciefx, ciefy = calculate_tool.rgb2cie(nir, red, blue)
    ndii = calculate_tool.get_ndii(nir, swir1)
    eri = calculate_tool.get_eri(nir, swir2)
    ndrbi = calculate_tool.get_ndrbi(blue, red)
    ndnbi = calculate_tool.get_ndnbi(blue, nir)
    ndswi = calculate_tool.get_swir12(swir1, swir2)
    ndwi = calculate_tool.get_ndwi(green, nir)
    ndsi = calculate_tool.get_ndsi(green, swir1)
    ndgi1 = calculate_tool.get_guano_index1(
        blue, green, red, nir, swir1, swir2)
    ndgi2 = calculate_tool.get_guano_index2(green, nir, swir1, swir2)

    bands_data = np.stack((blue, green, red, nir, swir1, swir2,
                           cietx, ciety, ciefx, ciefy,
                           ndii, eri, ndrbi, ndnbi, ndswi, ndwi, ndsi, ndgi1, ndgi2), axis=0)

    bands_data_y = bands_data.reshape(bands_data.shape[0], -1).T

    # y_pred = rf.predict(bands_data_y)
    # y_pred = y_pred.reshape(green.shape)

    # y_pred_xgb = xgbc.predict(bands_data_y)
    # y_pred_xgb = y_pred_xgb.reshape(green.shape)

    fig, axes = plt.subplots(1, 1, figsize=(18, 5))
    original_image = np.stack((nir, green, blue), axis=-1)
    axes.imshow(original_image)
    axes.set_title('Original Image')
    axes.axis('off')

    # marked_image = np.zeros(
    #     (green.shape[0], green.shape[1], 3), dtype=float)
    # marked_image[y_pred == 0] = [1, 0, 0]
    # marked_image[y_pred == 1] = [0, 1, 0]
    # marked_image[y_pred == 2] = [0, 0, 1]
    # marked_image[mask == 1] = [0, 1, 1]

    # xgb_image = np.zeros(
    #     (green.shape[0], green.shape[1], 3), dtype=float)
    # xgb_image[y_pred_xgb == 0] = [1, 0, 0]
    # xgb_image[y_pred_xgb == 1] = [0, 1, 0]
    # xgb_image[y_pred_xgb == 2] = [0, 0, 1]
    # # xgb_image[mask == 1] = [0, 1, 1]

    # axes[1].imshow(marked_image)
    # axes[1].set_title('Random Forest')
    # # axes[1].axis('on')
    # # axes[1].title('Marked Gray Scale Image with Red for y=0')
    # # legend_elements = [
    # #     Patch(facecolor=[1, 0, 0], edgecolor='white', label='guano- Red'),
    # #     Patch(facecolor=[0, 1, 0], edgecolor='white',
    # #           label='light part (snow and ice) - Green'),
    # #     Patch(facecolor=[0, 0, 1], edgecolor='white',
    # #           label='dark part (water and rock) - Blue'),
    # #     Patch(facecolor=[0, 1, 1], edgecolor='white', label='rock mask - Cyan')
    # # ]

    # # axes[1].legend(handles=legend_elements,
    # #                bbox_to_anchor=(1.05, 1), loc='upper left')

    # axes[2].imshow(xgb_image)
    # axes[2].set_title('XGBoost')
    # legend_elements = [
    #     Patch(facecolor=[1, 0, 0], edgecolor='white', label='guano- Red'),
    #     Patch(facecolor=[0, 1, 0], edgecolor='white',
    #           label='light part (snow and ice) - Green'),
    #     Patch(facecolor=[0, 0, 1], edgecolor='white',
    #           label='dark part (water and rock) - Blue'),
    #     Patch(facecolor=[0, 1, 1], edgecolor='white', label='rock mask - Cyan')
    # ]

    # axes[2].legend(handles=legend_elements,
    #                bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.tight_layout()
    plt.show()
