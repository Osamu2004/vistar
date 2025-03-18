# modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/registry.py
import logging



__all__ = ['Registry', 'AUGMENTATION','LOSS','DATAPROVIDER','OPT']


def _register_generic(module_dict, module_name, module, override=False):
    module_name = module_name if module_name else module.__name__
    if not override:
        if module_name in module_dict:
            logging.warning('{} has been in module_dict.'.format(module_name))
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    3): used as decorator when declaring the module named via __name__:
        @some_registry.register()
        def foo():
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name=None, module=None, override=False):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module, override)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn, override)
            return fn

        return register_fn

def register_dir(dir_name):
    import importlib
    import os

    # 遍历指定目录下的所有文件
    for root, dirs, files in os.walk(dir_name):
        print(f"Processing directory: {root}")  # 打印当前处理的目录
        if os.path.basename(root).startswith('_'):  # 跳过以 `_` 开头的目录
            continue

        # 替换文件路径分隔符为 '.'，以符合 Python 模块路径规则
        prefix = root.replace('/', '.').replace('\\', '.').strip('.')

        # 筛选 .py 文件并排除以 `_` 开头的文件
        py_files = [f for f in files if f.endswith('.py') and not f.startswith('_')]
        for pyf in py_files:
            module_name = f"{prefix}.{pyf.replace('.py', '')}"  # 构造模块名
            print(f"Loading module: {module_name}")  # 打印模块名
            try:
                importlib.import_module(module_name)  # 动态导入模块
            except ModuleNotFoundError as e:
                print(f"ModuleNotFoundError: {e}")
            except Exception as e:
                print(f"Unexpected error while loading module {module_name}: {e}")





AUGMENTATION = Registry()
LOSS = Registry()
DATAPROVIDER = Registry()
OPT = Registry()
MODEL = Registry()
