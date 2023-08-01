import isaacgym

assert isaacgym
import torch

import glob
import pickle as pkl
import lcm
import os



def load_and_save_policy(run_path, label):

    import wandb
    api = wandb.Api()
    run = api.run(run_path)

    # load config
    from dribblebot.envs.base.legged_robot_config import Cfg
    from dribblebot.envs.go1.go1_config import config_go1
    config_go1(Cfg)

    all_cfg = run.config
    cfg = all_cfg["Cfg"]

    for key, value in cfg.items():
        if hasattr(Cfg, key):
            for key2, value2 in cfg[key].items():
                setattr(getattr(Cfg, key), key2, value2)

    from dribblebot import MINI_GYM_ROOT_DIR
    path = MINI_GYM_ROOT_DIR + "/runs/" + run_path + "/" + label + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    print("path: ", path)

    adaptation_module_path = os.path.join(path, f'adaptation_module.jit')
    adaptation_module_file = wandb.restore('tmp/legged_data/adaptation_module_latest.jit', run_path=run_path)
    # adaptation_module_file = run.file('tmp/legged_data/adaptation_module_68000.jit').download(replace=True, root='./tmp')
    adaptation_module = torch.jit.load(adaptation_module_file.name, map_location="cpu")
    adaptation_module.save(adaptation_module_path)

    body_path = os.path.join(path, f'body.jit')
    body_file = wandb.restore('tmp/legged_data/body_latest.jit', run_path=run_path)
    # body_file = run.file('tmp/legged_data/body_68000.jit').download(replace=True, root='./tmp')
    body = torch.jit.load(body_file.name, map_location="cpu")
    body.save(body_path)

    ac_weights_path = os.path.join(path, f'ac_weights.pt')
    ac_weights_file = wandb.restore('tmp/legged_data/ac_weights_68000.pt', run_path=run_path)
    # ac_weights_file = run.file('tmp/legged_data/ac_weights_68000.pt').download(replace=True, root='./tmp')
    ac_weights = torch.load(ac_weights_file.name, map_location="cpu")
    # ac_weights.load_state_dict(torch.load(ac_weights_file.name))
    torch.save(ac_weights, ac_weights_path)

    cfg_path = os.path.join(path, f'config.yaml')
    import yaml
    with open(cfg_path, 'w') as f:
        yaml.dump(all_cfg, f)

if __name__ == '__main__':

    run_path = "improbableailab/dribbling/bvggoq26"
    label = "dribbling_pretrained"
    

    load_and_save_policy(run_path, label)