import os
import importlib

current_dir = os.path.dirname(__file__)

for f in os.listdir(current_dir):
    # 只处理 .py 文件
    if not f.endswith('.py'):
        continue

    # 跳过 __init__.py
    if f == '__init__.py':
        continue

    module = os.path.splitext(f)[0]

    # Linux 关键防护：模块名必须是合法标识符
    if not module.isidentifier():
        continue

    importlib.import_module(f'{__name__}.{module}')