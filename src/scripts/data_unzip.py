from pathlib import Path
import sys

# 添加项目根目录到sys.path
src_root = Path(__file__).parent.parent
sys.path.append(str(src_root))

try:
    from tools.file_tool import data_unzip
except ModuleNotFoundError as e:
    print(f"❌ 导入失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 输入路径：存放zip文件的目录
    input_path = r"D:\Sentinel2_data\zips"

    # 输出路径：存放结果的目录
    output_path = r"D:\Sentinel2_data\output"

    # 调用处理函数
    data_unzip(input_path, output_path, keep_temp=False)
