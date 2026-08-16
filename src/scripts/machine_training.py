from pathlib import Path
import joblib
import shutil
import sys

src_root = Path(__file__).parent.parent
sys.path.append(str(src_root))

try:
    from tools.training_tool import grid_search, load_config
except ModuleNotFoundError as e:
    print(f"❌ 导入失败: {e}")


def main():
    # ========== 1. 路径配置 ==========
    data_path = Path("data")
    config_path = Path("config/test.yaml")
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # ========== 2. 加载配置 ==========
    config = load_config(config_path)
    print(f"模型类型: {config['model_type']}")
    print(f"随机种子: {config['randnseed']}")
    print(f"采样数: {config['sample_nums']}")

    # ========== 3. 运行网格搜索 ==========
    print("\n=== 开始网格搜索 ===\n")
    results = grid_search(
        data_path=data_path,
        config=config,
        save_dir='output/grid_search',
        prefix=config['model_type'],
    )

    # ========== 4. 选出最佳模型（优先 cv_accuracy，其次 accuracy） ==========
    best = max(results, key=lambda r: r.get('cv_accuracy', r['accuracy']))

    # ========== 5. 保存最佳模型 ==========
    best_source_path = Path(best['save_path'])
    best_target_path = model_dir / f"{config['model_type']}_best.joblib"
    shutil.copy(best_source_path, best_target_path)

    print(f"\n 最佳模型已保存: {best_target_path}")

    # ========== 6. 输出最佳参数 ==========
    print("\n" + "="*50)
    print(" 最佳模型参数")
    print("="*50)
    print(f"参数组合: {best['combo_params']}")
    print(f"训练集精度: {best['accuracy']:.4f}")
    print(f"交叉验证精度: {best['cv_accuracy']:.4f}")
    print(f"交叉验证 Precision: {best['cv_precision']:.4f}")
    print(f"交叉验证 Recall: {best['cv_recall']:.4f}")
    print(f"交叉验证 F1: {best['cv_f1']:.4f}")
    print(f"保存路径: {best_target_path}")
    print("="*50)


if __name__ == "__main__":
    main()
