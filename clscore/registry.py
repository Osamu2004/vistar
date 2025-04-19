from apps.registry import register_dir

def register_seg_dataloaders():
    register_dir('segcore/data_provider')



def register_cd_augment():
    register_dir('apps/augment')

def register_cd_opt():
    register_dir('apps/trainer')

def register_cd_loss():
    register_dir('apps/trainer')

def register_cd_weight_init():
    register_dir('model')

def register_seg_all():
    register_seg_dataloaders()
    register_cd_augment()
    register_cd_opt()
    register_cd_loss()
    register_cd_weight_init()
