import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from apps import setup
from apps.utils.misc import  parse_unknown_args,dump_config
from clscore.trainer.cls_trainer import  ClsTrainer
from clscore.trainer.cls_run_config import  ClsRunConfig
from clscore.registry import register_cls_all

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
parser.add_argument('--offline', action='store_true', help='Enable wandb offline mode') # 解析
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

def main():
    # parse args
    register_cls_all()
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # setup gpu and distributed training
    #setup.setup_dist_env(args.gpu)

    # setup path, update args, and save args to path
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # setup random seed
    setup.setup_seed(args.manual_seed, args.resume)

    # setup exp config
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)

    # save exp config
    setup.save_exp_config(config, args.path)
    logger = setup.init_logger(logger_type="wandb",project=args.project,config = config,save_dir = args.path,name=args.name,offline=args.offline)
    # setup data provider
    data_provider = setup.setup_data_provider(config, is_distributed=False)

    # setup run config
    run_config = setup.setup_run_config(config, ClsRunConfig)

    # setup model
    model = create_efficientvit_cls_model(config["net_config"]["name"], False, dropout=config["net_config"]["dropout"])
    apply_drop_func(model.backbone.stages, config["backbone_drop"])

    # setup trainer
    trainer = ClsTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
        auto_restart_thresh=args.auto_restart_thresh,
    )
    # initialization
    setup.init_model(
        trainer.network,
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    # prep for training
    trainer.prep_for_training(run_config, config["ema_decay"], args.amp)

    # resume
    if args.resume:
        trainer.load_model()
        trainer.data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=True)
    else:
        trainer.sync_model()

    # launch training
    trainer.train(save_freq=args.save_freq)


if __name__ == "__main__":
    main()