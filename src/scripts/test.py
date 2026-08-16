import geopandas as gpd
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

if __name__ == '__main__':
    shp_path = Path(r'data')

    shp_files = list(shp_path.glob('*.shp'))
    gdf_list = []

    for idx, shp_path in enumerate(shp_files, start=1):
        gdf_temp = gpd.read_file(shp_path)
        gdf_temp['group_id'] = idx
        gdf_list.append(gdf_temp)

    # ========== 合并所有数据 ==========
    gdf_combined = pd.concat(gdf_list, ignore_index=True)

    # ========== 交叉表：Id × group_id ==========
    cross_tab = pd.crosstab(gdf_combined['group_id'], gdf_combined['Id'])

    print(cross_tab.to_string())
