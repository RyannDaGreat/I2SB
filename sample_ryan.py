# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug

import rp
from icecream import ic

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    print(f"compute_batch: corrupt_type={corrupt_type}")
    if "inpaint" in corrupt_type:
        assert False, 'Ryan: I dont use things with masks'
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    else:
        # clean_img, y = out
        clean_img, y = out, None

        rp.fansi_print(f"compute_batch: clean_img = {clean_img}",'blue','bold')
        # assert isinstance(clean_img, str), clean_img
        clean_img = rp.download_files_to_cache(clean_img)
        clean_img = rp.load_images(clean_img, use_cache=True)
        clean_img = rp.cv_resize_images(clean_img,size=(256,256))
        clean_img = rp.as_float_images(clean_img)
        clean_img = rp.as_rgb_images(clean_img)
        clean_img = rp.as_torch_images(clean_img)
        clean_img = clean_img * 2 - 1

        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        x1 = corrupt_img.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, x1, mask, cond, y

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # # build imagenet val dataset
    # val_dataset = build_val_dataset(opt, log, corrupt_type)
    val_dataset = [
        'https://upload.wikimedia.org/wikipedia/commons/6/67/Kim_Petras_%2842743719761%29.jpg',
        # 'https://media.npr.org/assets/img/2023/07/06/kim-petras---credit-luke-gilford_wide-8bbaafdd1a30bd518d7da2791a81179de2099127.jpg',
        # 'https://static.wikia.nocookie.net/kim-petras/images/1/12/F5XuxfnbwAAgu8L.jpg/revision/latest?cb=20231117000449',
    ]
    val_dataset = rp.get_all_image_files("/efs/users/ryan.burgert/public/I2SB_With_Checkpoints/downloaded_datasets/videos/bear/frames")
    val_dataset = rp.get_all_image_files("/efs/users/ryan.burgert/public/I2SB_With_Checkpoints/downloaded_datasets/videos/kevin_spin/frames/")
    n_samples = len(val_dataset)


    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    ic(opt)

    # create save folder
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    recon_imgs = []
    corrupt_imgs = []
    ys = []
    num = 0

    assert opt.global_rank == 0, 'RYAN: Please only run this with ONE gpu at a time to sync frames with the loaded noise.'

    for loader_itr, out in enumerate(val_loader):

        corrupt_img, x1, mask, cond, y = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)


        assert y is None, 'Ryan: I dont care about labels'
        del y
        corrupt_imgs.append(corrupt_img)

        xs, _ = runner.ddpm_sampling(
            ckpt_opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node==1
        )
        recon_img = xs[:, 0, ...].to(opt.device)

        assert recon_img.shape == corrupt_img.shape

        if loader_itr == 0 and opt.global_rank == 0: # debug
            os.makedirs(".debug", exist_ok=True)
            tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png")
            tu.save_image((recon_img+1)/2, ".debug/recon.png")
            log.info("Saved debug images!")

        # [-1,1]
        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)

        # y = y.to(opt.device)
        # gathered_y = collect_all_subset(y, log)
        # ys.append(gathered_y)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        dist.barrier()

    del runner

    arr         = torch.cat(recon_imgs  , axis=0)#[:n_samples]
    arr_corrupt = torch.cat(corrupt_imgs, axis=0)#[:n_samples]

    ic(arr.shape, arr_corrupt.shape)
    # label_arr = torch.cat(ys, axis=0)[:n_samples]

    if opt.global_rank == 0:
        # torch.save({"arr": arr, "label_arr": label_arr}, recon_imgs_fn)
        torch.save({"arr": arr}, recon_imgs_fn)
        log.info(f"Save at {recon_imgs_fn}")

        vis_images_folder = rp.make_directory(rp.get_parent_folder(str(recon_imgs_fn)) + "/image_files")

        vis_images = rp.as_numpy_array(
            rp.horizontally_concatenated_videos(
                rp.as_numpy_images(arr_corrupt),
                rp.as_numpy_images(arr),
            )
        )
        vis_images = vis_images / 2 + 0.5

        saved_paths = rp.save_images(
            vis_images,
            vis_images_folder,
        )

        log.info(f"Saved images:{saved_paths}")

    dist.barrier()

    # log.info(f"Sampling complete! Collect recon_imgs={arr.shape}, ys={label_arr.shape}")
    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
