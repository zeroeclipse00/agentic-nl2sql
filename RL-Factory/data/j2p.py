import os
import json
import pandas as pd

def convert_json_to_parquet():
    """
    自动扫描当前目录下所有 .json 文件
    逐个转换成同名的 .parquet 文件
    支持：单行JSON、JSON数组、标准JSON格式
    """
    # 获取当前脚本所在目录
    current_dir = os.getcwd()
    print(f"📂 当前目录：{current_dir}\n")

    # 遍历所有文件
    json_files = [f for f in os.listdir(current_dir) if f.endswith(".json")]

    if not json_files:
        print("❌ 当前目录没有找到任何 .json 文件")
        return

    print(f"✅ 找到 {len(json_files)} 个 JSON 文件，开始转换...\n")

    for filename in json_files:
        json_path = os.path.join(current_dir, filename)
        parquet_filename = os.path.splitext(filename)[0] + ".parquet"
        parquet_path = os.path.join(current_dir, parquet_filename)

        try:
            # 读取 JSON（兼容 99% 场景）
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 转 DataFrame → Parquet
            df = pd.DataFrame(data)
            df.to_parquet(parquet_path, index=False)

            print(f"✅ 成功：{filename} → {parquet_filename}")

        except Exception as e:
            print(f"❌ 失败：{filename}，错误：{str(e)}")

    print("\n🎉 全部转换完成！")

if __name__ == "__main__":
    convert_json_to_parquet()
