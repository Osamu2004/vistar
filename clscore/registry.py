from apps.registry import register_dir

def register_cls_dataloaders():
    register_dir('clscore/data_provider')



def register_cls_augment():
    register_dir('apps/augment')

def register_cls_opt():
    register_dir('apps/trainer')

def register_cls_loss():
    register_dir('apps/trainer')

def register_cls_weight_init():
    register_dir('model')

def register_cls_all():
    register_cls_dataloaders()
    register_cls_augment()
    register_cls_opt()
    register_cls_loss()
    register_cls_weight_init()
