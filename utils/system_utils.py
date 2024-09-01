#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

# def searchForMaxIteration(folder):
#     saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
#     return max(saved_iters)

# import os
import re

def searchForMaxIteration(folder):
    # 编译一个正则表达式，用于匹配文件名中的数字
    pattern = re.compile(r'chkpnt_(\d+)\.pth')
    max_iter = -1  # 初始化最大迭代次数为-1

    # 列出文件夹中的所有文件
    for fname in os.listdir(folder):
        # 使用正则表达式搜索文件名中的迭代次数
        match = pattern.search(fname)
        if match:
            # 如果找到匹配项，将匹配到的数字转换为整数
            iter_number = int(match.group(1))
            # 更新最大迭代次数
            max_iter = max(max_iter, iter_number)

    # 返回找到的最大迭代次数，如果没有找到任何匹配项，则返回None
    return max_iter if max_iter != -1 else None