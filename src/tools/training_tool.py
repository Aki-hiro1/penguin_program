from pathlib import Path
from itertools import product
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import joblib
import sys
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.base import clone

# 你的内部工具
from . import calculate_tool


# DATA_PROCESSING_MAPPING = {
#     0: lambda dn: dn,
#     1: lambda dn: np.hstack((dn, calculate_tool.indices_generate(dn))),
# }

def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"配置文件为空或格式错误: {config_path}")

    # ========== 1. 转换 sample_nums: list → dict ==========
    sample_nums_list = cfg.get('sample_nums', [])
    if isinstance(sample_nums_list, list):
        sample_nums = {i: int(n) for i, n in enumerate(sample_nums_list)}
    else:
        sample_nums = {int(k): int(v) for k, v in sample_nums_list.items()}

    # ========== 2. 转换 mapping: 确保 key 和 value 都是 int ==========
    mapping = cfg.get('mapping', {})
    class_mapping = {int(k): int(v) for k, v in mapping.items()}

    # ========== 3. 模型类型字符串处理 ==========
    model_type = cfg.get('model', 'random_forest')
    if model_type == 'rf':
        model_type = 'random_forest'
    model_type = model_type.lower()

    # ========== 4. 构建返回结果 ==========
    config = {
        'band_info': cfg.get('band_info', []),
        'band_name': cfg.get('band_name', []),
        'index_name': cfg.get('index_name', []),
        'sample_nums': sample_nums,
        'class_mapping': class_mapping,
        'model_type': model_type,
        'pixel_info': cfg.get('pixel_info', []),
        'randnseed': cfg.get('randnseed', 42),
        'param_grid': cfg.get('param_grid', {}),
    }

    # print(" 配置加载完成:")
    # print(f"   模型: {config['model_type']}")
    # print(f"   波段数: {len(config['band_name'])}")
    # print(f"   类别数: {len(config['sample_nums'])}")
    # print(f"   随机种子: {config['randnseed']}")
    # print(f"   超参数组数: {len(config['param_grid'])}")

    return config


def load_data(data_path, BAND_NAMES, PIXEL_INFO):
    shp_files = list(data_path.glob('*.shp'))
    data_dn_n_list = []
    data_info_ilst = []
    data_group_id_list = []

    for idx, shp_path in enumerate(shp_files, start=1):
        gdf_temp = gpd.read_file(shp_path)
        gdf_temp['group_id'] = idx

        data_dn = gdf_temp[BAND_NAMES].values
        data_dn_normal = data_dn / 10000
        data_info = gdf_temp[PIXEL_INFO].values
        # data_group_id = gdf_temp['group_id'].values

        data_dn_n_list.append(data_dn_normal)
        data_info_ilst.append(data_info)
        # data_group_id_list.append(data_group_id)

    # data_shp = gpd.read_file(data_path)
    # data_dn = data_shp[BAND_NAMES].values
    # data_dn_normal = data_dn / 10000
    # data_info = data_shp[PIXEL_INFO].values
    # return data_dn_normal, data_info
    return data_dn_n_list, data_info_ilst


def random_sample(data, info, SAMPLE_NUMS, RANDSEED, CLASS_MAPPING):
    np.random.seed(RANDSEED)
    all_id = np.unique(info[:, 0])
    selected_indexes = []
    for i in all_id:
        current_indexes = np.where(info[:, 0] == i)[0]

        if len(current_indexes) >= SAMPLE_NUMS[i]:
            chosen = np.random.choice(
                current_indexes, SAMPLE_NUMS[i], replace=False)
        else:
            print(
                f"(id = {i}) has less than {SAMPLE_NUMS[i]} samples")
            chosen = current_indexes

        selected_indexes.extend(chosen)

    selected_indexes = np.array(selected_indexes)
    sampled_data = data[selected_indexes]
    sampled_info = info[selected_indexes].copy()
    original_ids = sampled_info[:, 0].astype(int)
    mapped_ids = np.array(
        [CLASS_MAPPING.get(int(id_val), int(id_val)) for id_val in original_ids])
    sampled_info[:, 0] = mapped_ids
    return sampled_data, sampled_info


def traing_sample_prepare(data_path, BAND_NAMES, PIXEL_INFO, SAMPLE_NUMS, RANDSEED, CLASS_MAPPING):
    data_dn_n_list, data_info_list = load_data(
        data_path, BAND_NAMES, PIXEL_INFO)

    sample_data_list = []
    sample_info_list = []

    for i in range(len(data_dn_n_list)):
        temp_data, temp_info = random_sample(
            data_dn_n_list[i], data_info_list[i], SAMPLE_NUMS, RANDSEED, CLASS_MAPPING)

        temp_data = np.hstack(
            (temp_data, calculate_tool.indices_generate(temp_data)))

        sample_data_list.append(temp_data)
        sample_info_list.append(temp_info)

    sample_data, sample_info = np.vstack(
        sample_data_list), np.vstack(sample_info_list)
    return sample_data, sample_info


def _compute_metrics(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred)

    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        return {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
    else:
        return {
            'confusion_matrix': cm.tolist(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
        }


def _average_fold_results(fold_results):
    keys = ['accuracy', 'precision', 'recall', 'f1']
    avg = {}
    for key in keys:
        values = [r[key] for r in fold_results]
        avg[f'avg_{key}'] = np.mean(values)
        avg[f'std_{key}'] = np.std(values)

    if 'tp' in fold_results[0]:
        for key in ['tp', 'tn', 'fp', 'fn']:
            avg[f'avg_{key}'] = np.mean([r[key] for r in fold_results])

    return avg


def grid_search(data_path, config, save_dir='output/grid_search', prefix='model'):
    # ========== 从 config 解包所有参数 ==========
    BAND_NAMES = config['band_name']
    PIXEL_INFO = config['pixel_info']
    SAMPLE_NUMS = config['sample_nums']
    CLASS_MAPPING = config['class_mapping']
    MODEL_TYPE = config['model_type']
    RANDSEED = config['randnseed']
    param_grid = config['param_grid']

    # ========== 1. 准备数据 ==========
    print("正在加载数据...")
    X, y_info = traing_sample_prepare(
        data_path=data_path,
        BAND_NAMES=BAND_NAMES,
        PIXEL_INFO=PIXEL_INFO,
        SAMPLE_NUMS=SAMPLE_NUMS,
        RANDSEED=RANDSEED,
        CLASS_MAPPING=CLASS_MAPPING,
    )
    y = y_info[:, 0].astype(int)
    group_ids = y_info[:, 1].astype(int)
    n_classes = len(np.unique(y))
    print(f"数据加载完成: X.shape={X.shape}, 类别数={n_classes}")

    # ========== 2. 准备超参数组合 ==========
    if not param_grid:
        if MODEL_TYPE == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
            }
        elif MODEL_TYPE == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
            }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    total = len(combinations)
    print(f"共 {total} 组超参数组合待测试")

    # ========== 3. 遍历参数组合 ==========
    results = []
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GroupKFold

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{total}] 测试参数: {params}")

        # 3.1 创建并训练模型（用全量数据）
        if MODEL_TYPE == 'xgboost':
            model = XGBClassifier(
                **params,
                random_state=RANDSEED,
                eval_metric='mlogloss',
                n_jobs=-1,
            )
        else:
            model = RandomForestClassifier(
                **params,
                random_state=RANDSEED,
                n_jobs=-1,
            )
        model.fit(X, y)

        # 3.2 计算训练集精度
        y_pred = model.predict(X)
        train_acc = accuracy_score(y, y_pred)

        # 3.3 交叉验证（和参考代码完全一样：传已训练好的模型）
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # cv_scores = cross_validate(
        #     model, X, y,
        #     cv=cv,
        #     scoring=['accuracy', 'precision_macro',
        #              'recall_macro', 'f1_macro'],
        # )
        gkf = GroupKFold(n_splits=5)
        cv_scores = cross_validate(
            model, X, y,
            cv=gkf,
            groups=group_ids,
            scoring=['accuracy', 'precision_macro',
                     'recall_macro', 'f1_macro'],
        )
        cv_accuracy = float(np.mean(cv_scores['test_accuracy']))
        cv_precision = float(np.mean(cv_scores['test_precision_macro']))
        cv_recall = float(np.mean(cv_scores['test_recall_macro']))
        cv_f1 = float(np.mean(cv_scores['test_f1_macro']))

        # 3.4 记录结果（字段名和参考代码完全一致）
        result = {
            'combo_idx': i,
            'combo_params': params,
            'model': model,
            'accuracy': train_acc,
            'cv_accuracy': cv_accuracy,
            'cv_precision': cv_precision,
            'cv_recall': cv_recall,
            'cv_f1': cv_f1,
        }
        results.append(result)

        print(
            f"  train_acc={train_acc:.4f}, "
            f"cv_acc={cv_accuracy:.4f}, cv_f1={cv_f1:.4f}"
        )

    # ========== 4. 导出汇总 CSV ==========
    df = pd.DataFrame([
        {
            'combo_idx': r['combo_idx'],
            'params': str(r['combo_params']),
            'train_accuracy': r['accuracy'],
            'cv_accuracy': r['cv_accuracy'],
            'cv_precision': r['cv_precision'],
            'cv_recall': r['cv_recall'],
            'cv_f1': r['cv_f1'],
        }
        for r in results
    ])
    df.to_csv(save_dir / 'grid_search_summary.csv', index=False)
    print(f"\n✅ 结果汇总已保存: {save_dir / 'grid_search_summary.csv'}")

    # ========== 5. 选出并保存最佳模型 ==========
    best = max(results, key=lambda r: r.get('cv_accuracy', r['accuracy']))
    best_model_path = save_dir / f"{prefix}_best.joblib"
    joblib.dump(best['model'], best_model_path)
    best['save_path'] = str(best_model_path)

    print(f"\n🏆 最佳模型 (按 cv_accuracy):")
    print(f"   参数: {best['combo_params']}")
    print(f"   train_acc: {best['accuracy']:.4f}")
    print(f"   cv_acc: {best['cv_accuracy']:.4f}")
    print(f"   cv_f1: {best['cv_f1']:.4f}")
    print(f"   保存路径: {best_model_path}")

    return results
