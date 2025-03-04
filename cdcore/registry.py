from apps.registry import register_dir

def register_cd_dataloaders():
    register_dir('cdcore/data_provider')



def register_cd_augment():
    register_dir('apps/augment')

def register_cd_opt():
    register_dir('apps/trainer')

def register_cd_loss():
    register_dir('apps/trainer')

def register_cd_all():
    register_cd_dataloaders()
    register_cd_augment()
    register_cd_opt()
    register_cd_loss()
