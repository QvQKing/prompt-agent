import os
import re
import sys
from typing import Iterable, List, Tuple, Optional

# ---------- 可配置项 ----------
INCLUDE_EXTS = {
    ".py", ".txt", ".md", ".rst", ".json", ".yaml", ".yml",
    ".ini", ".cfg", ".toml", ".sh", ".bat",
    ".csv", ".tsv", ".log",
}
EXCLUDE_DIR_NAMES = {
    ".git", ".hg", ".svn", ".idea", ".vscode",
    "__pycache__", "venv", ".venv", "env", ".env",
    "node_modules", "dist", "build", ".mypy_cache", ".ruff_cache",
}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB 上限，避免误扫巨型文件
CASE_INSENSITIVE = True                 # 是否大小写不敏感
USE_REGEX = False                       # 是否把 target_text 当作正则
REGEX_FLAGS = re.MULTILINE              # 正则 flags（默认多行）
# ---------------------------------

def seems_text(bytes_sample: bytes) -> bool:
    """
    粗略判断是否为文本文件：
    - 含有 \x00 视为二进制
    - 控制字符占比过高视为二进制
    """
    if not bytes_sample:
        return True
    if b"\x00" in bytes_sample:
        return False
    # 允许常见的制表/换行/回车
    text_ctrl = set(b"\t\n\r")
    # 0x20(空格)~0x7E(可见 ASCII) 以及常用空白符算“可读”
    readable = sum(1 for b in bytes_sample if (32 <= b <= 126) or (b in text_ctrl))
    ratio = readable / len(bytes_sample)
    return ratio > 0.85

def iter_candidate_files(root_dir: str, include_exts: Iterable[str], exclude_file: str) -> Iterable[str]:
    exclude_file_abs = os.path.abspath(exclude_file)
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
        # 过滤目录
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
        for name in filenames:
            path = os.path.join(dirpath, name)
            # 排除当前脚本
            if os.path.abspath(path) == exclude_file_abs:
                continue
            # 跳过过大的文件
            try:
                if os.path.getsize(path) > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            # 扩展名匹配 或 无扩展名也尝试
            _, ext = os.path.splitext(name)
            if ext.lower() in include_exts or ext == "":
                yield path

def read_text_safely(path: str, sample_bytes: int = 4096) -> Optional[str]:
    """
    尝试文本检测 + 容错读取：
    1) 先读前 sample_bytes 字节判断是否像文本
    2) 优先 utf-8 读取；失败则用 latin-1 兜底
    """
    try:
        with open(path, "rb") as fb:
            head = fb.read(sample_bytes)
            if not seems_text(head):
                return None
        # 真正读取
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()
    except Exception:
        return None

def compile_pattern(target_text: str):
    if USE_REGEX:
        flags = REGEX_FLAGS | (re.IGNORECASE if CASE_INSENSITIVE else 0)
        return re.compile(target_text, flags)
    else:
        # 普通字符串匹配时，做大小写预处理
        return target_text.lower() if CASE_INSENSITIVE else target_text

def find_matches_in_text(text: str, pattern, lineno_start: int = 1) -> List[Tuple[int, str]]:
    matches = []
    if USE_REGEX:
        for i, line in enumerate(text.splitlines(), start=lineno_start):
            if pattern.search(line):
                matches.append((i, line.rstrip()))
    else:
        needle = pattern  # 已根据 CASE_INSENSITIVE 预处理
        for i, line in enumerate(text.splitlines(), start=lineno_start):
            hay = line.lower() if CASE_INSENSITIVE else line
            if needle in hay:
                matches.append((i, line.rstrip()))
    return matches

def find_files_with_text(root_dir: str, target_text: str, exclude_file: str) -> List[Tuple[str, int, str]]:
    matched_results: List[Tuple[str, int, str]] = []
    pattern = compile_pattern(target_text)

    for path in iter_candidate_files(root_dir, INCLUDE_EXTS, exclude_file):
        content = read_text_safely(path)
        if content is None:
            continue
        try:
            for lineno, line in find_matches_in_text(content, pattern):
                matched_results.append((path, lineno, line))
        except Exception as e:
            print(f"匹配时出错 {path}: {e}")
    return matched_results

if __name__ == "__main__":
    # === 修改为你的目标目录与搜索文本 ===
    directory = "/data/yichao/Agent-R1"
    search_text = "You may call one or more functions to assist with the"
    # =================================
    current_file = sys.argv[0]  # 当前脚本路径（用于排除自身）

    results = find_files_with_text(directory, search_text, current_file)

    if results:
        print("找到以下匹配结果：")
        for file_path, lineno, line in results:
            print(f"{file_path} (第 {lineno} 行): {line}")
    else:
        print("未找到包含目标文本的文件。")
