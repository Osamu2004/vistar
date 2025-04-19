import argparse
import os
from apps.utils.misc import  parse_unknown_args,dump_config
from apps import setup
from segcore.registry import register_seg_all
from apps.registry import DATAPROVIDER,AUGMENTATION
from segcore.trainer.seg_run_config import SegRunConfig
from segcore.seg_model_zoo import create_seg_model
from segcore.trainer.seg_trainer import SegTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")
parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
parser.add_argument("--gpu", type=str, default=None)  # used in single machine experiments
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--amp", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")

# initialization
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)

parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
parser.add_argument("--save_freq", type=int, default=1)

parser.add_argument('--project', type=str, help='WandB project name', required=False, default='default_project')
parser.add_argument('--name', type=str, help='WandB run name', required=False, default='default_project')

import torch
def main():
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    register_seg_all()
    print("CUDA available:", torch.cuda.is_available())
    print("Registered DataProviders:", DATAPROVIDER)
    print("Registered AUGMENTATION:", AUGMENTATION)
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)
    #setup.setup_dist_env(args.gpu)
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))
    setup.setup_seed(args.manual_seed, args.resume)
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    setup.save_exp_config(config, args.path)
    logger = setup.init_logger(logger_type="wandb",project=args.project,config = config,save_dir = args.path,name=args.name)
    
    print(config)
    data_provider = setup.setup_data_provider(config, is_distributed=False)
    print(data_provider,2222222222222222222222)
    run_config = setup.setup_run_config(config, SegRunConfig)

    model = create_seg_model(config["net_config"]["name"],dataset = config['data_provider'].get('type'))
    print(model)

    trainer = SegTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
        auto_restart_thresh=args.auto_restart_thresh,
        logger = logger,
    )
    #setup.init_model(
    #  trainer.network,
    #    rand_init=args.rand_init,
    #    last_gamma=args.last_gamma,
    #)
    #param_mean = {name: p.mean().item() for name, p in model.named_parameters()}
    #for name, mean in param_mean.items():
        #print(f"Mean of {name}: {mean:.6f}")
    trainer.prep_for_training(run_config, config["ema_decay"], args.amp)
    if args.resume:
        trainer.load_model()
        #trainer.data_provider = setup.setup_data_provider(config, is_distributed=False)
    trainer.train(save_freq=args.save_freq)
    print("Hello, World!")
    logger.finish()

    #setup.setup_dist_env(args.gpu)S
if __name__ == "__main__":
    main()