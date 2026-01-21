import os
import multiprocessing
import opencc
from tqdm import tqdm

# --- 配置 ---
RAW_DATA_PATH = "./data/sutras/cbeta-text"
CLEANED_DATA_PATH = "./data/sutras/cbeta-text-cleaned"
NUM_CORES = multiprocessing.cpu_count()

def process_single_file(file_info):
    src_path, dest_path = file_info
    try:
        # 如果目标文件已存在且大小不为0，跳过（断点续传逻辑）
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return True
            
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        converter = opencc.OpenCC('t2s')
        
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(converter.convert(content))
        return True
    except Exception as e:
        print(f"❌ 错误: {src_path} -> {e}")
        return False

def run_etl():
    tasks = []
    print(f"--- 🔍 正在扫描原始文件... ---")
    for root, _, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith(".txt"):
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, RAW_DATA_PATH)
                dest_file = os.path.join(CLEANED_DATA_PATH, rel_path)
                tasks.append((src_file, dest_file))

    print(f"--- 🚀 启动 CPU 多进程转换 (并发: {NUM_CORES}) ---")
    with multiprocessing.Pool(NUM_CORES) as pool:
        list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks)))
    print(f"--- ✅ ETL 完成！简体文本存于: {CLEANED_DATA_PATH} ---")

if __name__ == "__main__":
    run_etl()