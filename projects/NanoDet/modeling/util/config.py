from .yacs import CfgNode

cfg_s = CfgNode(new_allowed=True)
cfg_s.save_dir = "./"
# common params for NETWORK
cfg_s.model = CfgNode()
cfg_s.model.arch = CfgNode(new_allowed=True)
cfg_s.model.arch.backbone = CfgNode(new_allowed=True)
cfg_s.model.arch.fpn = CfgNode(new_allowed=True)
cfg_s.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg_s.data = CfgNode(new_allowed=True)
cfg_s.data.train = CfgNode(new_allowed=True)
cfg_s.data.val = CfgNode(new_allowed=True)
cfg_s.device = CfgNode(new_allowed=True)
# train
cfg_s.schedule = CfgNode(new_allowed=True)

# logger
cfg_s.log = CfgNode()
cfg_s.log.interval = 50

# testing
cfg_s.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg_s.defrost()
    cfg_s.merge_from_file(args_cfg)
    cfg_s.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg_s, file=f)
