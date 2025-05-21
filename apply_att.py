#
# Copyright 2024 Kenji Harada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# This program is a Python script of the adaptive tensor tree (ATT) method
# to construct a generative model for a dataset. The details of the ATT
# method are in our preprint, arXiv:2408.10669.
#
# Aug 2024
#
# Kenji Harada
# Graduate School of Informatics, Kyoto University, Japan
#

import math
import numpy as np
import born_machine
import argparse
import os
import time
import torch
import copy
import re

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-b", "--batch", action="store_true", help="set batch mode in learning"
)
parser.add_argument(
    "-i", "--batch_interval", type=int, default=100, help="set batch interval"
)
parser.add_argument("-N", "--NSTEP", type=int, default=10000, help="Num. of updates")
parser.add_argument(
    "-D", "--dir", type=str, default=".", help="a directory to save results"
)
parser.add_argument(
    "--data_dir", type=str, default=".", help="a directory to load data files"
)
parser.add_argument(
    "-u",
    "--update_length",
    type=int,
    default=1000,
    help="The length of updates",
)
parser.add_argument(
    "-I",
    "--CONVERGENCE_CHECK_INTERVAL",
    type=int,
    default=1.0,
    help="Interval ratio to check a convergence",
)
parser.add_argument(
    "-P",
    "--CONVERGENCE_PRECISION",
    type=float,
    default=1e-6,
    help="relative precision to check a convergence",
)
parser.add_argument(
    "-T",
    "--CHECKOUT_TIME",
    type=float,
    default=1800.0,
    help="checkout time",
)
parser.add_argument(
    "--float",
    action="store_true",
    help="use float32",
)
parser.add_argument("-g", "--gpu", action="store_true", help="use GPU")
parser.add_argument("--simple_output", action="store_true", help="output only log")
parser.add_argument("-o", "--output", action="store_true", help="print a result")
parser.add_argument(
    "DATA_NAME",
    type=str,
    default="data",
    help="data name",
)
parser.add_argument(
    "TYPE",
    type=int,
    default=1,
    help="0: MPS, 1:Tree, 2:2D tree, 3:MPS with randomized pixels, 4:2D tree with randomized pixels, 5: Random tree, 6:snake-like MPS for 2D",
)
parser.add_argument(
    "STRATEGY",
    type=int,
    default=0,
    help="one_update:  0: one-site, 1: two-site, 2: two-site + reconnection (NLL), 3: two-site + reconnection (EMI), 4: one-site + reconnection (NLL), 5: one-site + reconnection (EMI)",
)
parser.add_argument(
    "ALGORITHM",
    type=int,
    default=0,
    help="one_move:  0:time + nlink, 1:time + random",
)
parser.add_argument("ALPHA", type=float, default=0.1, help="learning rate")
parser.add_argument("MAX_ITERATION", type=int, default=10, help="learning steps")
parser.add_argument(
    "DIM", type=int, default=10, help="upper dimension of tensor's index"
)
parser.add_argument(
    "NSAMPLE", type=int, default=1000, help="Num. of samples in one learning"
)
parser.add_argument("SEED", type=int, default=1234, help="random number's seed")
parser.add_argument(
    "XSEED",
    type=int,
    default=1234,
    help="random seed for a born machine",
)
args = parser.parse_args()

# set BASE_ID
BASE_ID = "bm_{}_TY{}_ST{}_ALG{}_ALP{}_MAX{}_D{}_NS{}_SE{}_XS{}".format(
    args.DATA_NAME,
    args.TYPE,
    args.STRATEGY,
    args.ALGORITHM,
    args.ALPHA,
    args.MAX_ITERATION,
    args.DIM,
    args.NSAMPLE,
    args.SEED,
    args.XSEED,
)
if args.batch:
    BASE_ID += "_BI{}".format(args.batch_interval)
if args.gpu:
    BASE_ID += "_G"
# check
if os.path.exists(args.dir) is False:
    os.mkdir(args.dir)
if os.path.isfile(args.dir + "/" + BASE_ID + ".end") or os.path.isfile(
    args.dir + "/" + BASE_ID + ".stop"
):
    quit(0)
if args.float:
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_dtype(torch.float64)
# read description file of data
description_file = f"{args.data_dir}/{args.DATA_NAME}.des"
data_description = dict()
with open(description_file, "r") as f:
    for line in f:
        if re.match("#", line):
            continue
        else:
            m = re.match(r"\s*(\S*)\s*=\s*(\S*)", line)
            if m:
                data_description[m.group(1)] = m.group(2)

if "sample_name" in data_description:
    sample_file = (
        f"{args.data_dir}/{args.DATA_NAME}_{data_description['sample_name']}.dat"
    )
else:
    sample_file = f"{args.data_dir}/{args.DATA_NAME}_sample.dat"
if "test_name" in data_description:
    test_file = f"{args.data_dir}/{args.DATA_NAME}_{data_description['test_name']}.dat"
else:
    test_file = f"{args.data_dir}/{args.DATA_NAME}_test.dat"
if "size" in data_description:
    image_size = int(data_description["size"])
    image_length = int(math.sqrt(image_size))
else:
    print("There is no image_size in the description file")
    exit(-1)
if "dtype" in data_description:
    data_dtype = data_description["dtype"]
else:
    data_dtype = "f4"
# load born machine and data
if os.path.isfile(args.dir + "/" + BASE_ID + ".pickle"):
    with open(args.dir + "/" + BASE_ID + ".pickle", "rb") as f:
        bm = born_machine.binary_tree.load(f)
        if not hasattr(bm, "dir"):
            bm.dir = args.dir
        rng = np.random.default_rng(args.SEED)
        sample_data = np.fromfile(sample_file, dtype=data_dtype)
        sample_data = np.reshape(sample_data, (-1, image_size, 2))
        sample_data = torch.tensor(sample_data)
        if args.batch:
            bm.batch_data = sample_data
        else:
            nunits = max(int(sample_data.shape[0] / args.NSAMPLE), 1)
            target = rng.integers(nunits)
            target_data = sample_data[
                target * args.NSAMPLE : (target + 1) * args.NSAMPLE, :, :
            ]
else:
    rng = np.random.default_rng(args.SEED)
    # set data
    sample_data = np.fromfile(sample_file, dtype=data_dtype)
    sample_data = np.reshape(sample_data, (-1, image_size, 2))
    sample_data = torch.tensor(sample_data)
    if not args.batch:
        nunits = max(int(sample_data.shape[0] / args.NSAMPLE), 1)
        target = rng.integers(nunits)
        target_data = sample_data[
            target * args.NSAMPLE : (target + 1) * args.NSAMPLE, :, :
        ]
    # make a bm
    ## TYPE
    new_order = None
    if args.TYPE == 2:
        new_order = np.zeros(image_size, dtype=np.int64)
        for ip in range(image_size):
            ix = int(ip % image_length)
            iy = int(ip / image_length)
            ipx = 0
            base = 1
            nbits = int.bit_length(image_length - 1)
            for i in range(nbits):
                ipx += int(base * (ix % 2))
                ix = ix // 2
                base = base * 2
                ipx += int(base * (iy % 2))
                iy = iy // 2
                base = base * 2
            new_order[ipx] = ip
        GTYPE = 1
    elif args.TYPE == 3:
        new_order = rng.permutation(image_size)
        GTYPE = 0
    elif args.TYPE == 4:
        new_order = rng.permutation(image_size)
        GTYPE = 1
    elif args.TYPE == 5:
        GTYPE = 2
    elif args.TYPE == 6:
        new_order = np.zeros(image_size, dtype=np.int64)
        for ip in range(image_size):
            ix = int(ip % image_length)
            iy = int(ip / image_length)
            if iy % 2 == 0:
                ipx = ip
            else:
                ipx = iy * image_length + (image_length - ix - 1)
            new_order[ipx] = ip
        GTYPE = 0
    else:
        GTYPE = args.TYPE

    if args.gpu:
        bm = born_machine.binary_tree(
            image_size, GTYPE, BASE_ID, args.dir, args.XSEED, "cuda"
        )
    else:
        bm = born_machine.binary_tree(
            image_size, GTYPE, BASE_ID, args.dir, args.XSEED, "cpu"
        )
    # Which is a better initialization?
    if args.STRATEGY == 1 or args.STRATEGY == 2 or args.STRATEGY == 3:
        bm.set_tensor(2)
    else:
        bm.set_tensor(args.DIM)

    bm.new_order = new_order
    if new_order is not None:
        bm.permute_leaf(new_order)
    # set data into a bm
    if not args.batch:
        bm.set_data(target_data)
    else:
        bm.set_batch_data(sample_data, args.NSAMPLE, 0, args.batch_interval)
    bm.scan()

# load test data
test_data = np.fromfile(test_file, dtype=data_dtype)
test_data = np.reshape(test_data, (-1, image_size, 2))
test_data = torch.tensor(test_data)


# start learning
def output_log(bm):
    if len(bm.log) > 0:
        with open(args.dir + "/" + bm.base_id + ".log", "a") as f:
            f.write(bm.log)
            bm.log = ""


def bm_dump(bm, m, reduced=False):
    if reduced:
        bm2 = copy.copy(bm)
        bm2.tensor = None
        bm2.center_weight = None
        bm2.edge_weight = None
        bm2.logscale_edge_weight = None
        bm2.rng = None
        bm2.batch_data = None
        with open(m, "wb") as new_f:
            bm2.dump(new_f)
    else:
        with open(m, "wb") as f:
            bm.dump(f)


start_counter = bm.counter
status = 0
prev_counter = bm.counter
start_time = time.perf_counter()
while bm.counter < args.NSTEP:
    status = bm.sweep(
        args.update_length - (bm.counter - prev_counter),
        args.ALPHA,
        args.DIM,
        args.STRATEGY,
        args.ALGORITHM,
        output=args.output,
        max_iteration=args.MAX_ITERATION,
        conv_check_interval=args.CONVERGENCE_CHECK_INTERVAL,
        conv_check_precision=args.CONVERGENCE_PRECISION,
        checkout_time=args.CHECKOUT_TIME,
    )
    output_log(bm)
    if status == 0:
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > args.CHECKOUT_TIME:
            if not bm.checkout:
                bm_dump(bm, args.dir + "/" + bm.base_id + ".pickle")
                with open(args.dir + "/" + bm.base_id + ".history", "a") as f:
                    f.write("checkout: {} {}\n".format(bm.sweep_counter, bm.counter))
            start_time = time.perf_counter()
        prev_counter = bm.counter
        bm.set_data(test_data)
        (loss, r) = bm.calc_loss()
        with open(args.dir + "/" + bm.base_id + "_test.log", "a") as f:
            f.write(
                "{:d} {:d} {:g} {:g}\n".format(bm.counter, bm.sweep_counter, loss, r)
            )
        if args.batch:
            bm.recover_current_batch_data()
        else:
            bm.set_data(target_data)
        continue
    elif status == 1 or status == 2:
        break
    elif status == 5:
        with open(args.dir + "/" + bm.base_id + ".history", "a") as f:
            f.write(
                "There is no valid loss after the svd of a two-site update: {}\n".format(
                    bm.counter
                )
            )
        if args.batch:
            bm.set_next_batch_data(force=True)
            continue
        else:
            bm_dump(bm, args.dir + "/" + bm.base_id + "_stop.pickle")
            quit(0)
    elif status == 6:
        bm_dump(bm, args.dir + "/" + bm.base_id + ".pickle")
        with open(args.dir + "/" + bm.base_id + ".history", "a") as f:
            f.write("loop: {} {}\n".format(bm.sweep_counter, bm.counter))
        bm.set_data(test_data)
        (loss, r) = bm.calc_loss()
        with open(args.dir + "/" + bm.base_id + "_test.log", "a") as f:
            f.write(
                "{:d} {:d} {:g} {:g}\n".format(bm.counter, bm.sweep_counter, loss, r)
            )
        if args.batch:
            bm.recover_current_batch_data()
        else:
            bm.set_data(target_data)

if status == 0 or status == 1:
    bm_dump(bm, args.dir + "/" + bm.base_id + ".pickle")
    with open(args.dir + "/" + bm.base_id + ".history", "a") as f:
        if status == 0:
            f.write("reach to NSTEP\n")
        elif status == 1:
            f.write("loss convergence\n")
    with open(args.dir + "/" + bm.base_id + ".end", "w") as f:
        if bm.counter == start_counter:
            speed = None
        else:
            speed = (time.perf_counter() - start_time) / (bm.counter - start_counter)
        output = ""
        for x in (
            args.TYPE,
            args.STRATEGY,
            args.ALGORITHM,
            args.ALPHA,
            args.MAX_ITERATION,
            args.DIM,
            args.NSAMPLE,
            args.SEED,
            args.XSEED,
        ):
            output += " {}".format(x)
        if args.batch:
            output += " {}".format(args.batch_interval)
        f.write(
            "{} {}".format(
                bm.calc_loss(),
                speed,
            )
            + output
            + "\n"
        )
else:
    bm_dump(
        bm,
        args.dir
        + "/"
        + bm.base_id
        + "_LP{}_{}.pickle".format(bm.sweep_counter, bm.counter),
    )
