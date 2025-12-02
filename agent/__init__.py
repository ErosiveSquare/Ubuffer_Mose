# /your_project_root/agent/__init__.py

from .er import *
from .scr import *
from .mose import *
from .joint import *
from .buf import *
from .ablation_mose import AblationMOSE_v3 # <--- 添加这一行

METHODS = {
    'er': ER,
    'scr': SCR,
    'mose': MOSE,
    'joint': Joint,
    'buf': Buf,
    'ablation_mose_v3': AblationMOSE_v3, # <--- 添加这一行
}

def get_agent(method_name, *args, **kwargs):
    if method_name in METHODS.keys():
        return METHODS[method_name](*args, **kwargs)
    else:
        raise Exception('unknown method!')
