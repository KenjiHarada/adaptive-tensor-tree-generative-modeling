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
# This Python code is for the adaptive tensor tree (ATT) method
# to construct a generative model for a dataset. The details of the ATT
# method are in our preprint, arXiv:2408.10669.
#
# Aug 2024
#
# Kenji Harada
# Graduate School of Informatics, Kyoto University, Japan
#

import numpy as np
import opt_einsum as oe
import math
import queue
from collections import deque
import time
import pickle
import os
import sys
import torch


class CalcError(Exception):
    pass


class binary_tree:
    # nleaf = -1  # int
    # nnode = -1  # int
    # nedge = -1  # int
    # center_edge = -1  # int
    # node_connection = None  # narray, np.int32
    # node_descendant = None  # narray, np.int32
    # edge_connection = None  # narray, np.int32
    # tensor = []  # list of torch.Tensor(cuda)
    # center_weight = None  # torch.Tensor(cuda)
    # edge_weight = []  # list of torch.Tensor(cuda)
    # logscale_edge_weight = None  # torch.Tensor(cuda)
    # edge_ee = None  # torch.Tensor
    # edge_mi = None  # torch.Tensor
    # rng = None  # (cuda)
    # rng_state = None

    # visit_time = None  # narray, np.uint32
    # nupdate_edge = None  # narray, np.uint32
    # counter = -1  # int
    # base_id = None
    # log = ""
    # sweep_counter = 0

    # batch_size = -1  # int
    # batch_point = -1  # int
    # batch_data = None
    # batch_interval = 1
    # batch_counter = 0

    def load(f):
        bm = pickle.load(f)
        bm.rng = torch.Generator(device=bm.base_device)
        bm.rng.set_state(bm.rng_state)
        bm.rng_state = None
        return bm

    def __init__(self, n, type, base_id, dir=".", seed=None, base_device=None):
        if base_device is None:
            self.base_device = "cpu"
        else:
            self.base_device = base_device
        self.rng = torch.Generator(device=self.base_device)
        self.rng.manual_seed(seed)
        self.create_structure(n, type)
        self.checked = np.zeros(self.nedge - self.nleaf)
        self.nuncheck = self.nedge - self.nleaf
        self.base_id = base_id
        self.dir = dir
        self.log = ""
        self.sweep_counter = 0
        self.previous_loss = 0
        self.accumurate_loss = 0
        self.batch_data = None

    def record_log(self, m, plus=False):
        print(
            "# " + m + "  ({}) in {}".format(self.counter, self.base_id),
            file=sys.stderr,
        )
        if plus:
            self.log += "# " + m + "\n"

    def create_structure(self, n, type):
        self.nleaf = n
        self.nnode = n - 2
        self.nedge = 2 * n - 3  # (3*(n-2) - n)/2 + n
        self.node_connection = np.zeros([self.nnode, 3], dtype=np.int32)
        self.edge_connection = np.full([self.nedge, 2], -1, dtype=np.int32)
        self.visit_time = np.zeros(
            [
                self.nedge,
            ],
            dtype=np.uint32,
        )
        self.nupdate_edge = np.zeros(
            [
                self.nedge,
            ],
            dtype=np.uint32,
        )
        self.counter = 0
        if type == 0:  # tensor train
            it = 0
            ie = self.nleaf
            self.node_connection[it, 0] = ie
            self.edge_connection[ie, 0] = 3 * it + 0

            self.node_connection[it, 1] = 0
            self.edge_connection[0, 1] = 3 * it + 1
            self.edge_connection[0, 0] = -1

            self.node_connection[it, 2] = 1
            self.edge_connection[1, 1] = 3 * it + 2
            self.edge_connection[1, 0] = -1
            it += 1
            for i in range(2, self.nleaf - 2):
                self.node_connection[it, 1] = ie
                self.edge_connection[ie, 1] = 3 * it + 1
                ie += 1

                self.node_connection[it, 2] = i
                self.edge_connection[i, 1] = 3 * it + 2
                self.edge_connection[i, 0] = -1

                self.node_connection[it, 0] = ie
                self.edge_connection[ie, 0] = 3 * it + 0
                it += 1

            self.node_connection[it, 0] = ie
            self.edge_connection[ie, 1] = 3 * it + 0
            self.center_edge = ie
            ie += 1

            i = self.nleaf - 2
            self.node_connection[it, 1] = i
            self.edge_connection[i, 1] = 3 * it + 1
            self.edge_connection[i, 0] = -1

            self.node_connection[it, 2] = i + 1
            self.edge_connection[i + 1, 1] = 3 * it + 2
            self.edge_connection[i + 1, 0] = -1
            it += 1
        elif type == 1:  # balance tree
            q = queue.Queue()
            for i in range(self.nleaf):
                q.put(i)
                self.edge_connection[i, 0] = -1
            it = 0
            ie = self.nleaf
            while q.qsize() > 3:
                e0 = q.get()
                e1 = q.get()
                self.node_connection[it, 1] = e0
                self.edge_connection[e0, 1] = 3 * it + 1
                self.node_connection[it, 2] = e1
                self.edge_connection[e1, 1] = 3 * it + 2
                self.node_connection[it, 0] = ie
                self.edge_connection[ie, 0] = 3 * it + 0
                q.put(ie)
                ie += 1
                it += 1
            e0 = q.get()
            e1 = q.get()
            self.node_connection[it, 1] = e0
            self.edge_connection[e0, 1] = 3 * it + 1
            self.node_connection[it, 2] = e1
            self.edge_connection[e1, 1] = 3 * it + 2
            e2 = q.get()
            self.node_connection[it, 0] = e2
            self.edge_connection[e2, 1] = 3 * it + 0
            self.center_edge = e2
            it += 1
        elif type == 2:  # random tree
            it = 0
            ie = self.nleaf
            for i in range(self.nleaf):
                self.edge_connection[i, 0] = -1
            edge_list = np.arange(self.nleaf)
            num = self.nleaf
            while num > 4:
                p0 = (
                    torch.randint(
                        num, (1,), generator=self.rng, device=self.base_device
                    )
                ).item()
                p1 = (
                    p0
                    + 1
                    + torch.randint(
                        (num - 1), (1,), generator=self.rng, device=self.base_device
                    ).item()
                ) % num
                if p0 > p1:
                    p0, p1 = p1, p0
                px = edge_list[p0]
                py = edge_list[p1]
                self.node_connection[it, 0] = ie
                self.node_connection[it, 1] = px
                self.node_connection[it, 2] = py
                self.edge_connection[ie, 0] = 3 * it + 0
                self.edge_connection[px, 1] = 3 * it + 1
                self.edge_connection[py, 1] = 3 * it + 2
                edge_list[p0] = ie
                if p1 < (num - 1):
                    edge_list[p1] = edge_list[num - 1]
                num -= 1
                it += 1
                ie += 1
            if self.base_device == "cuda":
                order = torch.randperm(
                    4, generator=self.rng, device=self.base_device
                ).cpu()
            else:
                order = torch.randperm(4, generator=self.rng, device=self.base_device)
            self.node_connection[it, 0] = ie
            self.node_connection[it, 1] = edge_list[order[0]]
            self.node_connection[it, 2] = edge_list[order[1]]
            self.edge_connection[ie, 0] = 3 * it
            self.edge_connection[edge_list[order[0]], 1] = 3 * it + 1
            self.edge_connection[edge_list[order[1]], 1] = 3 * it + 2
            it += 1
            self.node_connection[it, 0] = ie
            self.node_connection[it, 1] = edge_list[order[2]]
            self.node_connection[it, 2] = edge_list[order[3]]
            self.edge_connection[ie, 1] = 3 * it
            self.edge_connection[edge_list[order[2]], 1] = 3 * it + 1
            self.edge_connection[edge_list[order[3]], 1] = 3 * it + 2
            it += 1
            self.center_edge = ie
            ie += 1

        if ie != self.nedge or it != self.nnode:
            self.record_log("Error to make a tree.: {} {}".format(it, ie))
            exit(-1)

        self.setup_descendant()

        return

    def setup_descendant(self):
        if not hasattr(self, "node_descendant") or self.node_descendant is None:
            self.node_descendant = np.zeros([self.nnode, 3], dtype=np.int32)
        check_node = np.zeros([self.nnode], dtype=np.int8)
        q = queue.Queue()
        for ie in range(self.nleaf):
            ix = self.edge_connection[ie, 1]
            it = ix // 3
            i0 = ix % 3
            check_node[it] += 1
            self.node_descendant[it, i0] = 1
            if check_node[it] == 2:
                q.put(it)
        q0 = queue.Queue()
        while q.qsize() > 0:
            it = q.get()
            ie = self.node_connection[it, 0]
            if (self.edge_connection[ie, 0] // 3) == it:
                ip = 1
            else:
                ip = 0
            ix = self.edge_connection[ie, ip]
            it0 = ix // 3
            in0 = ix % 3
            if in0 != 0:
                check_node[it0] += 1
            else:
                q0.put(it0)
            self.node_descendant[it0, in0] = (
                self.node_descendant[it, 1] + self.node_descendant[it, 2]
            )
            if in0 != 0 and check_node[it0] == 2:
                q.put(it0)
        while q0.qsize() > 0:
            it = q0.get()
            for i in range(1, 3):
                ie = self.node_connection[it, i]
                if self.edge_connection[ie, 0] == -1:
                    continue
                if (self.edge_connection[ie, 0] // 3) == it:
                    ip = 1
                else:
                    ip = 0
                ix = self.edge_connection[ie, ip]
                it0 = ix // 3
                in0 = ix % 3
                if in0 != 0:
                    raise ValueError
                if in0 == 0:
                    self.node_descendant[it0, in0] = (
                        self.node_descendant[it, 0] + self.node_descendant[it, 3 - i]
                    )
                    q0.put(it0)
        return

    def permute_leaf(self, new_order):
        # new_order: np.array(int), size = nleaf, new_order[i] = j means j-th leaf connects to i-th leaf of a network
        node_list = np.zeros(self.nleaf, dtype=int)
        for i in range(self.nleaf):
            node_list[i] = self.edge_connection[i, 1]
        xperm = np.zeros(self.nleaf, dtype=int)
        for i in range(self.nleaf):
            xperm[new_order[i]] = i
        node_list = node_list[xperm]
        for i in range(self.nleaf):
            x = node_list[i]
            node = x // 3
            index = x % 3
            self.node_connection[node, index] = i
            self.edge_connection[i, 1] = x
        return

    def calc_ee(self, w):
        p = w * w
        p = p[p > 0]
        p /= torch.sum(p)
        return (-torch.sum(torch.log(p) * p)).item()

    def calc_mi(self):
        t = []
        w = []
        log_w_all = 0
        for i in range(2):
            it = self.edge_connection[self.center_edge, i] // 3
            t.append(self.tensor[it])
            for j in range(1, 3):
                ie = self.node_connection[it, j]
                w.append(self.edge_weight[ie])
                log_w_all += self.logscale_edge_weight[ie]
        # condition: |center_weight| = 1
        wx = self.center_weight / torch.linalg.vector_norm(self.center_weight)
        w0 = wx * wx
        q1 = oe.contract(
            "abc, ade, a, fb, fc, fd, fe -> f",
            t[0],
            t[1],
            wx,
            w[0],
            w[1],
            w[2],
            w[3],
        )
        px1 = oe.contract(
            "abc, a, ade, fb, fc, fd, fe -> f", t[0], w0, t[0], w[0], w[1], w[0], w[1]
        )
        py1 = oe.contract(
            "abc, a, ade, fb, fc, fd, fe -> f", t[1], w0, t[1], w[2], w[3], w[2], w[3]
        )
        mi2 = abs(
            torch.sum(
                2 * torch.log(torch.abs(q1)) - torch.log(px1) - torch.log(py1)
            ).item()
            / q1.numel()
        )
        return mi2

    def set_tensor(self, upper_dim):
        it0 = self.edge_connection[self.center_edge, 0] // 3
        it1 = self.edge_connection[self.center_edge, 1] // 3
        q = deque([])
        q.append(self.node_connection[it0, 1])
        q.append(self.node_connection[it0, 2])
        q.append(self.node_connection[it1, 1])
        q.append(self.node_connection[it1, 2])
        sorted_edges = deque([])

        while len(q) > 0:
            edge = q.popleft()
            sorted_edges.append(edge)
            if self.edge_connection[edge, 0] == -1:
                continue
            if self.edge_connection[edge, 0] % 3 == 0:
                it = self.edge_connection[edge, 0] // 3
            else:
                it = self.edge_connection[edge, 1] // 3
            q.append(self.node_connection[it, 1])
            q.append(self.node_connection[it, 2])

        edge_dim = np.full((self.nedge), -1, dtype=np.int32)
        for i in range(self.nleaf):
            edge_dim[i] = 2
        while len(sorted_edges) > 0:
            edge = sorted_edges.pop()
            if self.edge_connection[edge, 0] == -1:
                edge_dim[i] = 2
                continue
            if self.edge_connection[edge, 0] % 3 == 0:
                it = self.edge_connection[edge, 0] // 3
            else:
                it = self.edge_connection[edge, 1] // 3
            edim1 = edge_dim[self.node_connection[it, 1]]
            edim2 = edge_dim[self.node_connection[it, 2]]
            edge_dim[edge] = min(edim1 * edim2, upper_dim)

        edim0 = edge_dim[self.node_connection[it0, 1]]
        edim1 = edge_dim[self.node_connection[it0, 2]]
        edim2 = edge_dim[self.node_connection[it1, 1]]
        edim3 = edge_dim[self.node_connection[it1, 2]]
        edge_dim[self.center_edge] = min(edim0 * edim1, edim2 * edim3, upper_dim)

        self.tensor = []
        for i in range(self.nnode):
            xshape = (
                edge_dim[self.node_connection[i, 0]],
                edge_dim[self.node_connection[i, 1]],
                edge_dim[self.node_connection[i, 2]],
            )
            t = torch.rand(size=xshape, generator=self.rng, device=self.base_device)
            t = t.reshape((xshape[0], xshape[1] * xshape[2]))
            (u, s, vh) = torch.linalg.svd(t, full_matrices=False)
            new_t = vh.reshape(xshape)
            self.tensor.append(new_t)
        w = torch.ones(edge_dim[self.center_edge], device=self.base_device)
        self.center_weight = w / edge_dim[self.center_edge]

        self.edge_ee = torch.zeros(self.nedge)
        self.edge_ee[self.center_edge] = self.calc_ee(self.center_weight)
        self.edge_mi = torch.zeros(self.nedge)
        return

    def set_batch_data(
        self, batch_data, batch_size=100, batch_point=0, batch_interval=1
    ):
        self.batch_data = batch_data
        self.batch_size = batch_size
        self.batch_point = batch_point
        self.batch_interval = batch_interval
        self.batch_counter = 0
        target_data = self.batch_data[
            batch_point * batch_size : (batch_point + 1) * batch_size, :, :
        ]
        self.set_data(target_data)
        self.batch_point += 1

    def set_data(self, data):
        it0 = self.edge_connection[self.center_edge, 0] // 3
        it1 = self.edge_connection[self.center_edge, 1] // 3
        q = deque([])
        q.append(self.node_connection[it0, 1])
        q.append(self.node_connection[it0, 2])
        q.append(self.node_connection[it1, 1])
        q.append(self.node_connection[it1, 2])
        sorted_edges = deque([])

        while len(q) > 0:
            edge = q.popleft()
            sorted_edges.append(edge)
            if self.edge_connection[edge, 0] == -1:
                continue
            if self.edge_connection[edge, 0] % 3 == 0:
                it = self.edge_connection[edge, 0] // 3
            else:
                it = self.edge_connection[edge, 1] // 3
            q.append(self.node_connection[it, 1])
            q.append(self.node_connection[it, 2])

        if not data is None:
            self.logscale_edge_weight = torch.zeros(self.nedge, device=self.base_device)
            self.edge_weight = [None] * self.nedge
            if self.base_device == "cuda":
                if torch.get_default_dtype() == torch.float64:
                    for i in range(self.nleaf):
                        self.edge_weight[i] = ((data[:, i, :]).double()).cuda()
                else:
                    for i in range(self.nleaf):
                        self.edge_weight[i] = ((data[:, i, :]).float()).cuda()
            else:
                if torch.get_default_dtype() == torch.float64:
                    for i in range(self.nleaf):
                        self.edge_weight[i] = (data[:, i, :]).double()
                else:
                    for i in range(self.nleaf):
                        self.edge_weight[i] = (data[:, i, :]).float()
        else:
            self.logscale_edge_weight.fill_(0.0)
            for i in range(self.nleaf, self.nedge):
                self.edge_weight[i] = None

        while len(sorted_edges):
            edge = sorted_edges.pop()
            if not self.edge_weight[edge] is None:
                continue
            if self.edge_connection[edge, 0] % 3 == 0:
                it = self.edge_connection[edge, 0] // 3
            else:
                it = self.edge_connection[edge, 1] // 3
            w1 = self.edge_weight[self.node_connection[it, 1]]
            ls1 = self.logscale_edge_weight[self.node_connection[it, 1]]
            w2 = self.edge_weight[self.node_connection[it, 2]]
            ls2 = self.logscale_edge_weight[self.node_connection[it, 2]]
            new_w = oe.contract("ab, ac, dbc -> ad", w1, w2, self.tensor[it])
            new_ls = math.log(torch.max(torch.abs(new_w)).item())
            new_w /= math.exp(new_ls)
            self.edge_weight[edge] = new_w
            self.logscale_edge_weight[edge] = new_ls + ls1 + ls2
        return

    def calc_psi(self):
        def _calc_edge(ie):
            return (self.edge_weight[ie], self.logscale_edge_weight[ie])

        def _calc_node(it):
            (w1, ls1) = _calc_edge(self.node_connection[it, 1])
            (w2, ls2) = _calc_edge(self.node_connection[it, 2])
            new_w = oe.contract("ab, ac, dbc -> ad", w1, w2, self.tensor[it])
            new_ls = math.log(torch.max(torch.abs(new_w)).item())
            new_w /= math.exp(new_ls)
            return (new_w, new_ls + ls1 + ls2)

        it1 = self.edge_connection[self.center_edge, 0] // 3
        it2 = self.edge_connection[self.center_edge, 1] // 3
        (w1, ls1) = _calc_node(it1)
        (w2, ls2) = _calc_node(it2)
        psi = oe.contract("ab, ab, b -> a", w1, w2, self.center_weight)
        z = torch.max(torch.abs(psi)).item()
        if z == 0:
            ls0 = 0
        else:
            ls0 = math.log(z)
            psi /= math.exp(ls0)
        logscale_psi = ls0 + ls1 + ls2
        return (psi, logscale_psi)

    def check_zero_anomaly(self, x):
        flag = (x == 0) | torch.isnan(x) | torch.isinf(x)
        if torch.count_nonzero(flag) > 0:
            raise CalcError("Zero or anomaly")

    def calc_loss(self):
        (psi, logscale_psi) = self.calc_psi()
        r = 1.0
        zsqrt = torch.linalg.norm(self.center_weight)
        loss = (
            2
            * (
                -torch.sum(torch.log(torch.abs(psi))) / psi.numel()
                + torch.log(zsqrt)
                - logscale_psi
            ).item()
        )
        return loss, r

    def calc_grad(self):
        ts = []
        w = []
        w2 = []
        logscale_gpsi = 0
        for i1 in range(2):
            it = self.edge_connection[self.center_edge, i1] // 3
            ts.append(self.tensor[it])
            for i2 in range(1, 3):
                logscale_gpsi += self.logscale_edge_weight[self.node_connection[it, i2]]
                w.append(self.edge_weight[self.node_connection[it, i2]])
            w2.append(
                oe.contract(
                    "ab, ac, dbc -> ad", w[2 * i1], w[2 * i1 + 1], self.tensor[it]
                )
            )
        gpsi = oe.contract("ab, ac, ad, ae -> bcdea", w[0], w[1], w[2], w[3])
        psi = oe.contract("ab, ab, b -> a", w2[0], w2[1], self.center_weight)
        zsqrt = torch.linalg.norm(self.center_weight)
        t = oe.contract("abc, ade, a -> bcde", ts[0], ts[1], self.center_weight)
        try:
            self.check_zero_anomaly(psi)
        except CalcError:
            return (t, None, None, None)
        ratio = self._calc_ratio_in_grad(psi, gpsi)
        grad = (2.0 / (zsqrt * zsqrt)) * t + ratio
        return (t, grad, gpsi, logscale_gpsi)

    def calc_grad_tt(self, t, gpsi):
        psi = oe.contract("bcde, bcdea -> a", t, gpsi)
        self.check_zero_anomaly(psi)
        ratio = self._calc_ratio_in_grad(psi, gpsi)
        zsqrt = torch.linalg.norm(torch.ravel(t))
        grad = (2.0 / (zsqrt * zsqrt)) * t + ratio
        return grad

    def _calc_ratio_in_grad(self, psi, gpsi):
        x = torch.sum(gpsi / psi, axis=-1)
        if torch.count_nonzero(torch.isinf(x) | torch.isnan(x)) > 0:
            raise CalcError("Cannot calculate a ratio of gpsi and psi.")
        else:
            n = psi.numel()
            if n == 0:
                raise CalcError("Cannot calculate a ratio of gpsi and psi.")
            else:
                return -(2.0 / n) * x

    def svd(self, m):
        (u, s, vh) = torch.linalg.svd(m, full_matrices=False)
        if torch.count_nonzero(torch.isinf(s) | torch.isnan(s)) > 0:
            raise CalcError("s has error in svd")
        return (u, s, vh)

    def move_center(self, ipt, ipe):
        # ipt = 0 or 1
        # ipe = 1 or 2
        ## collect info
        ix = self.edge_connection[self.center_edge, ipt]
        itx = ix // 3
        ie = self.node_connection[itx, ipe]
        if self.edge_connection[ie, 0] == -1:
            self.record_log("Cannot move to leaf.")
            return False
        iy = self.edge_connection[ie, 0]
        if (iy // 3) == itx:
            iy = self.edge_connection[ie, 1]
        ity = iy // 3
        iz = self.edge_connection[self.center_edge, ipt ^ 1]
        itz = iz // 3
        ## update tensors and weights
        if ipe == 1:
            t0 = oe.contract("abc, a -> bac", self.tensor[itx], self.center_weight)
        elif ipe == 2:
            t0 = oe.contract("abc, a -> cba", self.tensor[itx], self.center_weight)

        xshape = t0.shape
        (u, new_center_weight, vh) = self.svd(t0.reshape((xshape[0], -1)))
        new_tx = torch.reshape(vh, (-1, xshape[1], xshape[2]))
        new_ty = oe.contract("abc, ae -> ebc", self.tensor[ity], u)
        iez1 = self.node_connection[itz, 1]
        iez2 = self.node_connection[itz, 2]
        new_edge_weight = oe.contract(
            "ab, ac, dbc -> ad",
            self.edge_weight[iez1],
            self.edge_weight[iez2],
            self.tensor[itz],
        )
        new_logscale = torch.log(torch.max(torch.abs(new_edge_weight))).item()
        new_edge_weight /= math.exp(new_logscale)
        new_logscale += (
            self.logscale_edge_weight[iez1] + self.logscale_edge_weight[iez2]
        )
        self.tensor[itx] = new_tx
        self.center_weight = new_center_weight / torch.linalg.norm(new_center_weight)
        self.tensor[ity] = new_ty
        self.edge_weight[self.center_edge] = new_edge_weight
        self.logscale_edge_weight[self.center_edge] = new_logscale
        ## update the information of connection

        def _set_edge_to(_ie, _it, _i):
            _ip = 0
            if (self.edge_connection[_ie, _ip] // 3) != _it:
                _ip = 1
            self.edge_connection[_ie, _ip] = _it * 3 + _i

        if ipe == 1:
            _set_edge_to(self.node_connection[itx, 0], itx, 1)
            _set_edge_to(self.node_connection[itx, 1], itx, 0)
            self.node_connection[itx, 0], self.node_connection[itx, 1] = (
                self.node_connection[itx, 1],
                self.node_connection[itx, 0],
            )
            self.node_descendant[itx, 0], self.node_descendant[itx, 1] = (
                self.node_descendant[itx, 1],
                self.node_descendant[itx, 0],
            )
        elif ipe == 2:
            _set_edge_to(self.node_connection[itx, 0], itx, 2)
            _set_edge_to(self.node_connection[itx, 2], itx, 0)
            self.node_connection[itx, 0], self.node_connection[itx, 2] = (
                self.node_connection[itx, 2],
                self.node_connection[itx, 0],
            )
            self.node_descendant[itx, 0], self.node_descendant[itx, 2] = (
                self.node_descendant[itx, 2],
                self.node_descendant[itx, 0],
            )
        self.center_edge = ie
        return True

    def update_center_pairs(self, alpha, t, s, w, logscale_w_all, nup=1):
        try:
            for iup in range(nup):
                for ipt in range(2):
                    iptx = ipt ^ 1
                    gpsi = oe.contract(
                        "ab, ac, dbc, ae, af -> defa",
                        w[iptx * 2],
                        w[iptx * 2 + 1],
                        t[iptx],
                        w[ipt * 2],
                        w[ipt * 2 + 1],
                    )
                    psi = oe.contract("bcda, bcd, b -> a", gpsi, t[ipt], s)
                    self.check_zero_anomaly(psi)
                    ratio = self._calc_ratio_in_grad(psi, gpsi)
                    zsqrt = torch.linalg.norm(s)
                    tx = oe.contract("abc, a -> abc", t[ipt], s)
                    grad = (2.0 / (zsqrt * zsqrt)) * tx + ratio
                    if torch.count_nonzero(torch.isnan(grad) | torch.isinf(grad)) > 0:
                        raise CalcError("Cannot calculate a grad for a single update.")
                    new_tx = tx - alpha * grad
                    xshape = new_tx.shape
                    (u, s, vh) = self.svd(new_tx.reshape(xshape[0], -1))
                    s /= torch.linalg.norm(s)
                    t[ipt] = torch.reshape(vh, (-1, xshape[1], xshape[2]))
                    t[iptx] = oe.contract("abc, ad -> dbc", t[iptx], u)
        except CalcError:
            raise CalcError("Cannot update a center pari.")
        psi = oe.contract(
            "ab, ac, ad, ae, fbc, f, fde -> a", w[0], w[1], w[2], w[3], t[0], s, t[1]
        )
        zsqrt = torch.linalg.norm(s)
        loss = (
            -torch.sum(torch.log(torch.abs(psi))) / psi.numel()
            + torch.log(zsqrt)
            - logscale_w_all
        ).item() * 2
        return (t, s, loss, False)

    def recover_current_batch_data(self):
        if self.batch_data is None:
            return
        if self.batch_point == 0:
            size = self.batch_data.shape[0]
            data = self.batch_data[size - self.batch_size - 1 :, :, :]
        else:
            self.batch_point -= 1
            i0 = self.batch_point * self.batch_size
            i1 = (self.batch_point + 1) * self.batch_size
            data = self.batch_data[i0:i1, :, :]
            self.batch_point += 1
        self.set_data(data)

    def set_next_batch_data(self, force=False):
        if not self.batch_data is None:
            if not force:
                if self.batch_interval == 0:
                    return
                elif self.batch_counter < (self.batch_interval - 1):
                    self.batch_counter += 1
                    return
            size = self.batch_data.shape[0]
            if (self.batch_point + 1) * self.batch_size > size:
                data = self.batch_data[size - self.batch_size - 1 :, :, :]
                self.batch_point = 0
            else:
                i0 = self.batch_point * self.batch_size
                i1 = (self.batch_point + 1) * self.batch_size
                data = self.batch_data[i0:i1, :, :]
                self.batch_point += 1
            self.set_data(data)
            self.batch_counter = 0

    def one_update(
        self,
        alpha,
        upper_dimension,
        strategy=0,
        max_iteration=1,
    ):
        # 0: one-site, 1: two-site, 2: two-site + reconnection
        do_two_sites_update = False
        do_reconnection = False
        do_reconnection_mi = False
        if strategy == 1:
            do_two_sites_update = True
        elif strategy == 2:
            do_two_sites_update = True
            do_reconnection = True
        elif strategy == 3:
            do_two_sites_update = True
            do_reconnection_mi = True
        elif strategy == 4:
            do_reconnection = True
        elif strategy == 5:
            do_reconnection_mi = True

        # define a local update routine from new_t with new_order
        def _update_pairs(t, w, log_w_all, new_order=None, nup=1):
            if new_order is None:
                tx = t
                new_order = range(4)
            else:
                tx = torch.permute(t, new_order)
            ty = tx.reshape(tx.shape[0] * tx.shape[1], tx.shape[2] * tx.shape[3])
            (u, s, vh) = self.svd(ty)
            if s.numel() > upper_dimension:
                u = u[:, :upper_dimension]
                s = s[:upper_dimension]
                vh = vh[:upper_dimension, :]
            ts = []
            ts.append(torch.reshape(u.t(), (-1, tx.shape[0], tx.shape[1])))
            ts.append(torch.reshape(vh, (-1, tx.shape[2], tx.shape[3])))
            ws = [w[new_order[0]], w[new_order[1]], w[new_order[2]], w[new_order[3]]]
            norm = torch.linalg.norm(ty)
            trunc = 1 - torch.sum(s * s) / (norm * norm)
            return (
                self.update_center_pairs(alpha, ts, s, ws, log_w_all, nup),
                trunc.item(),
            )

        # change a local structure
        def _update_edge(ie, ix0, ix1):
            for i in range(2):
                if self.edge_connection[ie, i] == ix0:
                    self.edge_connection[ie, i] = ix1
                    return
            self.record_log("ERROR in _update_edge.")
            quit(-1)

        # calculate mutual information_2
        def _calc_mi(t, s, w):
            # condition: |s| = 1
            s /= torch.linalg.norm(s)
            q1 = oe.contract(
                "abc, ade, a, fb, fc, fd, fe -> f",
                t[0],
                t[1],
                s,
                w[0],
                w[1],
                w[2],
                w[3],
            )
            w0 = s * s
            px1 = oe.contract(
                "abc, a, ade, fb, fc, fd, fe -> f",
                t[0],
                w0,
                t[0],
                w[0],
                w[1],
                w[0],
                w[1],
            )
            py1 = oe.contract(
                "abc, a, ade, fb, fc, fd, fe -> f",
                t[1],
                w0,
                t[1],
                w[2],
                w[3],
                w[2],
                w[3],
            )
            mi2 = abs(
                torch.sum(
                    2 * torch.log(torch.abs(q1)) - torch.log(px1) - torch.log(py1)
                ).item()
                / q1.numel()
            )
            return mi2

        # preparation
        rt = []
        rs = []
        ra = []
        w = []
        log_w_all = 0
        for i in range(2):
            it = self.edge_connection[self.center_edge, i] // 3
            for j in range(1, 3):
                w.append(self.edge_weight[self.node_connection[it, j]])
                log_w_all += self.logscale_edge_weight[self.node_connection[it, j]]
        descent = (
            self.node_descendant[self.edge_connection[self.center_edge, 0] // 3, 0]
            / self.nleaf
        )
        if descent > 0.5:
            descent = 1.0 - descent

        # two-site update
        if do_two_sites_update:
            (t, grad, gpsi, logscale_gpsi) = self.calc_grad()
            if grad is None:
                self.record_log("Cannot calculate a grad for a two-site update.")
                new_t = t
            else:
                new_t = t - alpha * grad
                for _ in range(max_iteration - 1):
                    try:
                        grad = self.calc_grad_tt(new_t, gpsi)
                    except CalcError:
                        break
                    else:
                        new_t = new_t - alpha * grad
            # reconnection
            if do_reconnection or do_reconnection_mi:
                nchoice = 3
                new_orders = [None, (0, 2, 1, 3), (0, 3, 2, 1)]
            else:
                nchoice = 1
                new_orders = [
                    None,
                ]
            rloss = np.zeros(nchoice)
            ree = np.zeros(nchoice)
            rmi = np.zeros(nchoice)
            truncs = np.zeros(nchoice)
            for i in range(nchoice):
                try:
                    ((t0, s0, loss0, a0), trunc) = _update_pairs(
                        new_t, w, log_w_all, new_orders[i], max_iteration
                    )
                except CalcError:
                    rt.append(None)
                    rs.append(None)
                    ra.append(False)
                else:
                    rt.append(t0)
                    rs.append(s0)
                    ra.append(not a0)
                    rloss[i] = loss0
                    ree[i] = self.calc_ee(s0)
                    if new_orders[i] is None:
                        rmi[i] = _calc_mi(t0, s0, w)
                    else:
                        rmi[i] = _calc_mi(
                            t0,
                            s0,
                            [
                                w[new_orders[i][0]],
                                w[new_orders[i][1]],
                                w[new_orders[i][2]],
                                w[new_orders[i][3]],
                            ],
                        )
                    truncs[i] = trunc
        else:
            # reconnection
            ts = []
            for i in range(2):
                it = self.edge_connection[self.center_edge, i] // 3
                ts.append(self.tensor[it])
            if do_reconnection or do_reconnection_mi:
                nchoice = 3
                new_orders = [None, (0, 2, 1, 3), (0, 3, 2, 1)]
            else:
                nchoice = 1
                new_orders = [
                    None,
                ]
            rloss = np.zeros(nchoice)
            ree = np.zeros(nchoice)
            rmi = np.zeros(nchoice)
            truncs = np.zeros(nchoice)
            if nchoice > 1:
                new_t = oe.contract(
                    "abc, ade, a -> bcde", ts[0], ts[1], self.center_weight
                )
            for i in range(nchoice):
                try:
                    if new_orders[i] is None:
                        (t0, s0, loss0, a0) = self.update_center_pairs(
                            alpha,
                            ts,
                            self.center_weight,
                            w,
                            log_w_all,
                            max_iteration,
                        )
                        trunc = 0e0
                    else:
                        ((t0, s0, loss0, a0), trunc) = _update_pairs(
                            new_t, w, log_w_all, new_orders[i], max_iteration
                        )
                except CalcError:
                    rt.append(None)
                    rs.append(None)
                    ra.append(False)
                else:
                    rt.append(t0)
                    rs.append(s0)
                    ra.append(not a0)
                    rloss[i] = loss0
                    ree[i] = self.calc_ee(s0)
                    if new_orders[i] is None:
                        rmi[i] = _calc_mi(t0, s0, w)
                    else:
                        rmi[i] = _calc_mi(
                            t0,
                            s0,
                            [
                                w[new_orders[i][0]],
                                w[new_orders[i][1]],
                                w[new_orders[i][2]],
                                w[new_orders[i][3]],
                            ],
                        )
                    truncs[i] = trunc

        # select new local connection
        ips = np.arange(nchoice)
        ip0 = ips[ra]
        rloss0 = rloss[ra]
        rloss_ave = np.average(rloss0)
        rloss_std = np.std(rloss0)
        ree0 = ree[ra]
        ree_ave = np.average(ree0)
        ree_std = np.std(ree0)
        rmi0 = rmi[ra]
        if len(ip0) == 0:
            self.record_log(
                "There is no valid loss after the svd of a two-site update."
            )
            status = 5
            return (status, None, None)
        else:
            if do_reconnection_mi:
                ip = ip0[np.argmin(rmi0)]
            else:
                ip = ip0[np.argmin(rloss0)]

        # local reconnection
        if ip == 1:
            it0 = self.edge_connection[self.center_edge, 0] // 3
            ie0 = self.node_connection[it0, 2]
            it1 = self.edge_connection[self.center_edge, 1] // 3
            ie1 = self.node_connection[it1, 1]
            _update_edge(ie0, it0 * 3 + 2, it1 * 3 + 1)
            _update_edge(ie1, it1 * 3 + 1, it0 * 3 + 2)
            self.node_connection[it0, 2] = ie1
            self.node_connection[it1, 1] = ie0
            self.node_descendant[it0, 2], self.node_descendant[it1, 1] = (
                self.node_descendant[it1, 1],
                self.node_descendant[it0, 2],
            )
            self.node_descendant[it1, 0] = (
                self.node_descendant[it0, 1] + self.node_descendant[it0, 2]
            )
            self.node_descendant[it0, 0] = (
                self.node_descendant[it1, 1] + self.node_descendant[it1, 2]
            )
        elif ip == 2:
            it0 = self.edge_connection[self.center_edge, 0] // 3
            ie0 = self.node_connection[it0, 2]
            it1 = self.edge_connection[self.center_edge, 1] // 3
            ie1 = self.node_connection[it1, 2]
            _update_edge(ie0, it0 * 3 + 2, it1 * 3 + 2)
            _update_edge(ie1, it1 * 3 + 2, it0 * 3 + 2)
            self.node_connection[it0, 2] = ie1
            self.node_connection[it1, 2] = ie0
            self.node_descendant[it0, 2], self.node_descendant[it1, 2] = (
                self.node_descendant[it1, 2],
                self.node_descendant[it0, 2],
            )
            self.node_descendant[it1, 0] = (
                self.node_descendant[it0, 1] + self.node_descendant[it0, 2]
            )
            self.node_descendant[it0, 0] = (
                self.node_descendant[it1, 1] + self.node_descendant[it1, 2]
            )
        for i in range(2):
            it = self.edge_connection[self.center_edge, i] // 3
            self.tensor[it] = rt[ip][i]
        self.center_weight = rs[ip]
        self.edge_ee[self.center_edge] = ree[ip]
        self.edge_mi[self.center_edge] = self.calc_mi()
        self.nupdate_edge[self.center_edge] += 1

        new_descent = (
            self.node_descendant[self.edge_connection[self.center_edge, 0] // 3, 0]
            / self.nleaf
        )
        if new_descent > 0.5:
            new_descent = 1.0 - new_descent

        # results
        status = 0
        return (
            status,
            (
                len(ip0),
                ip,
                rloss,
                ree,
                ra,
                rloss_ave,
                rloss_std,
                ree_ave,
                ree_std,
                max_iteration,
                descent,
                new_descent,
            ),
            rmi,
        )

    def one_move(self, algorithm=0):
        score = []
        edge_info = []
        check_mi_list = queue.Queue()
        for i in range(2):
            for j in range(1, 3):
                it = self.edge_connection[self.center_edge, i] // 3
                ie = self.node_connection[it, j]
                if self.edge_connection[ie, 0] == -1:
                    check_mi_list.put(2 * i + (j - 1))
                    if j == 1:
                        x = oe.contract(
                            "abc, a -> acb", self.tensor[it], self.center_weight
                        )
                    else:
                        x = oe.contract(
                            "abc, a -> abc", self.tensor[it], self.center_weight
                        )
                    xshape = x.shape
                    s = torch.linalg.svdvals(x.reshape((-1, xshape[2])))
                    self.edge_ee[ie] = self.calc_ee(s)
                    self.nupdate_edge[ie] += 1
                    self.visit_time[ie] = self.counter
                    continue

                score.append(self.visit_time[ie])
                edge_info.append((i, j))
        if not check_mi_list.empty():
            t = []
            w = []
            log_w_all = 0
            for i in range(2):
                it = self.edge_connection[self.center_edge, i] // 3
                t.append(self.tensor[it])
                for j in range(1, 3):
                    ie = self.node_connection[it, j]
                    w.append(self.edge_weight[ie])
                    log_w_all += self.logscale_edge_weight[ie]
            q1 = oe.contract(
                "abc, ade, a, fb, fc, fd, fe -> f",
                t[0],
                t[1],
                self.center_weight,
                w[0],
                w[1],
                w[2],
                w[3],
            )
            w0 = self.center_weight * self.center_weight
            while check_mi_list.qsize() > 0:
                ix = check_mi_list.get()
                i = ix // 2
                j = ix % 2 + 1
                if i == 0 and j == 1:
                    px1 = oe.contract(
                        "abc, a, adc, fb, fd -> f", t[0], w0, t[0], w[0], w[0]
                    )
                    py0 = oe.contract(
                        "abc, a, ade, fc, fd, fe -> bf",
                        t[0],
                        self.center_weight,
                        t[1],
                        w[1],
                        w[2],
                        w[3],
                    )
                elif i == 0 and j == 2:
                    px1 = oe.contract(
                        "abc, a, abd, fc, fd -> f", t[0], w0, t[0], w[1], w[1]
                    )
                    py0 = oe.contract(
                        "abc, a, ade, fb, fd, fe -> cf",
                        t[0],
                        self.center_weight,
                        t[1],
                        w[0],
                        w[2],
                        w[3],
                    )
                elif i == 1 and j == 1:
                    px1 = oe.contract(
                        "abc, a, adc, fb, fd -> f", t[1], w0, t[1], w[2], w[2]
                    )
                    py0 = oe.contract(
                        "abc, a, ade, fb, fc, fe -> df",
                        t[0],
                        self.center_weight,
                        t[1],
                        w[0],
                        w[1],
                        w[3],
                    )
                elif i == 1 and j == 2:
                    px1 = oe.contract(
                        "abc, a, abd, fc, fd -> f", t[1], w0, t[1], w[3], w[3]
                    )
                    py0 = oe.contract(
                        "abc, a, ade, fb, fc, fd -> ef",
                        t[0],
                        self.center_weight,
                        t[1],
                        w[0],
                        w[1],
                        w[2],
                    )
                py1 = oe.contract("xf, xf -> f", py0, py0)
                it = self.edge_connection[self.center_edge, i] // 3
                ie = self.node_connection[it, j]
                self.edge_mi[ie] = abs(
                    torch.sum(
                        2 * torch.log(torch.abs(q1)) - torch.log(px1) - torch.log(py1)
                    ).item()
                    / q1.numel()
                )

        xscore = np.array(score, dtype=int)
        ip = np.argmin(xscore)
        candidates = []
        for ip0 in range(xscore.size):
            if xscore[ip0] == xscore[ip]:
                candidates.append(ip0)
        if len(candidates) > 1:
            if algorithm == 0:
                ip = candidates[0]
            else:
                ip = candidates[
                    torch.randint(
                        0,
                        len(candidates),
                        1,
                        generator=self.rng,
                        device=self.base_device,
                    ).item()
                ]

        (ipt, ipe) = edge_info[ip]

        self.move_center(ipt, ipe)
        self.visit_time[self.center_edge] = self.counter
        self.counter += 1

    def scan(self):
        self.visit_time.fill(0)
        prev_counter = self.counter
        self.checked.fill(0)
        self.nuncheck = self.nedge - self.nleaf
        self.setup_descendant()
        while self.nuncheck > 0:
            self.edge_ee[self.center_edge] = self.calc_ee(self.center_weight)
            try:
                self.edge_mi[self.center_edge] = self.calc_mi()
            except CalcError:
                self.edge_mi[self.center_edge] = 0
            if self.checked[self.center_edge - self.nleaf] == 0:
                self.checked[self.center_edge - self.nleaf] = 1
                self.nuncheck -= 1
            self.one_move()
        self.visit_time.fill(0)
        self.counter = prev_counter
        self.checked.fill(0)
        self.nuncheck = self.nedge - self.nleaf

    def dump(self, f):
        if self.rng is None:
            self.rng_state = None
        else:
            self.rng_state = self.rng.get_state()
            self.rng = None
        if not self.batch_data is None:
            backup_data = self.batch_data.detach()
            self.batch_data = None
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.batch_data = backup_data
        else:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        if not self.rng_state is None:
            self.rng = torch.Generator(device=self.base_device)
            self.rng.set_state(self.rng_state)
            self.rng_state = None

    def sweep(
        self,
        nupdate,
        alpha,
        upper_dimension,
        strategy,
        algorithm,
        output=False,
        max_iteration=1,
        conv_check_interval=10.0,
        conv_check_precision=1e-5,
        checkout_time=1800.0,
    ):
        self.checkout = False
        start_counter = self.counter
        start_time = time.perf_counter()
        cci = int(conv_check_interval * (self.nedge - self.nleaf))
        while (self.counter - start_counter) < nupdate:
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > checkout_time:
                self.checkout = True
                with open(self.dir + "/" + self.base_id + ".pickle", "wb") as f:
                    self.dump(f)
                with open(self.dir + "/" + self.base_id + ".history", "a") as f:
                    f.write(
                        "checkout in sweep: {} {}\n".format(
                            self.sweep_counter, self.counter
                        )
                    )
                with open(self.dir + "/" + self.base_id + ".log", "a") as f:
                    f.write(self.log)
                    self.log = ""
                if self.counter != start_counter:
                    with open(self.dir + "/" + self.base_id + ".time", "a") as f:
                        f.write(
                            "{:g} {:d} {:d}\n".format(
                                elapsed_time / (self.counter - start_counter),
                                start_counter,
                                self.counter,
                            )
                        )
                if os.path.isfile(self.base_id + ".stop"):
                    self.record_log("There is a stop file.")
                    status = 2
                    return status
                start_time = time.perf_counter()

            try:
                (status, results, rmi) = self.one_update(
                    alpha,
                    upper_dimension,
                    strategy,
                    max_iteration,
                )
            except CalcError:
                pass
            else:
                if status != 0:
                    return status
                (
                    nip0,
                    ip,
                    rloss,
                    ree,
                    ra,
                    rloss_ave,
                    rloss_std,
                    ree_ave,
                    ree_std,
                    nstep_b,
                    descent,
                    new_descent,
                ) = results
                self.log += "{:d} {:g} {:g} {:d} {:d} {:d} {:g} {:g} {:g} {:g} {:d} {:g} {:g} {:g}\n".format(
                    self.counter,
                    rloss[ip],
                    ree[ip],
                    self.center_edge,
                    ip,
                    nip0,
                    rloss_ave,
                    rloss_std,
                    ree_ave,
                    ree_std,
                    nstep_b,
                    self.edge_mi[self.center_edge],
                    descent,
                    new_descent,
                )
                if output:
                    print(
                        "{:d} {:g} {:g} {:d} {:d} {:d} {:g} {:g} {:g} {:g} {:d} {:g} {:g}".format(
                            self.counter,
                            rloss[ip],
                            ree[ip],
                            self.center_edge,
                            ip,
                            nip0,
                            rloss_ave,
                            rloss_std,
                            ree_ave,
                            ree_std,
                            nstep_b,
                            self.edge_mi[self.center_edge],
                            descent,
                            new_descent,
                        ),
                        rmi,
                    )
                self.accumurate_loss += rloss[ip]
            # move
            if self.checked[self.center_edge - self.nleaf] == 0:
                self.checked[self.center_edge - self.nleaf] = 1
                self.nuncheck -= 1
            self.one_move(algorithm)
            if self.sweep_counter >= 5 and (self.counter % cci) == 0:
                x = self.accumurate_loss / cci
                if abs((x - self.previous_loss) / x) <= conv_check_precision:
                    self.record_log("Loss converges.", True)
                    status = 1
                    return status
                else:
                    self.previous_loss = x
                    self.accumurate_loss = 0
            # batch
            self.set_next_batch_data()

            if self.nuncheck == 0:
                if output:
                    self.record_log("finish sweep {}.".format(self.sweep_counter), True)
                self.sweep_counter += 1
                if not (self.batch_data is None) and self.batch_interval == 0:
                    self.set_next_batch_data(True)
                status = 6
                self.checked.fill(0)
                self.nuncheck = self.nedge - self.nleaf
                return status
        status = 0
        return status

    def sampling_set_data(
        self,
        nsample,
        data=None,
        single_site_marginal_distribution=False,
        reduced_leaves=None,
    ):
        """
        Variables:
            self.edge_matrix - edge_matrix[i] is torch.tensor(id_sample, k_up, k_down) at edge i
            self.logscale_edge_matrix - logscale_edge_matrix[i] is the logscale of edge_matrix[i][id_sample, k_up, k_down] at edge i
            self.sampled - sampled[i] is True if the leaf i is sampled
            self.nsampled - number of sampled leaves
            self.sampling_data - sampling_data[a, i] is the value of a sampling data at leaf i, a is sample number.

        Arguments:
            nsample - number of samples
            data    - list of (position, status), status is 0 or 1
            single_site_marginal_distribution - if True, the function set data for the single-site marginal distribution
        """
        self.edge_matrix = [None] * self.nedge
        self.logscale_edge_matrix = [None] * self.nedge
        self.sampled = torch.full(
            (self.nleaf,), False, dtype=torch.bool, device=self.base_device
        )
        self.nsampled = 0
        self.sampling_data = torch.zeros((nsample, self.nleaf), dtype=torch.uint8)

        # prepare
        it0 = self.edge_connection[self.center_edge, 0] // 3
        it1 = self.edge_connection[self.center_edge, 1] // 3
        q = deque([])
        q.append(self.node_connection[it0, 1])
        q.append(self.node_connection[it0, 2])
        q.append(self.node_connection[it1, 1])
        q.append(self.node_connection[it1, 2])
        sorted_edges = deque([])

        while len(q) > 0:
            edge = q.popleft()
            sorted_edges.append(edge)
            if self.edge_connection[edge, 0] == -1:
                continue
            if self.edge_connection[edge, 0] % 3 == 0:
                it = self.edge_connection[edge, 0] // 3
            else:
                it = self.edge_connection[edge, 1] // 3
            q.append(self.node_connection[it, 1])
            q.append(self.node_connection[it, 2])

        # set nleaf's data
        for i in range(self.nleaf):
            self.edge_matrix[i] = torch.zeros((nsample, 2, 2), device=self.base_device)
            self.edge_matrix[i][:, 0, 0] = 1
            self.edge_matrix[i][:, 1, 1] = 1
            self.logscale_edge_matrix[i] = torch.zeros(nsample, device=self.base_device)
        if not reduced_leaves is None:
            for pos in reduced_leaves:
                self.sampling_data[:, pos] = 2
                self.sampled[pos] = True
                self.nsampled += 1
        if not data is None:
            if single_site_marginal_distribution:
                for pos, status in data:
                    x = torch.flatten(self.edge_matrix[pos])
                    xpos = torch.arange(nsample) * 4 + (1 - status) * 2 + (1 - status)
                    x[xpos] = 0
                    self.sampling_data[:, pos] = status
                    self.sampled[pos] = True
                    self.nsampled += 1
            else:
                for pos, status in data:
                    self.edge_matrix[pos][:, 1 - status, 1 - status] = 0
                    self.sampling_data[:, pos] = status
                    self.sampled[pos] = True
                    self.nsampled += 1

        while len(sorted_edges):
            edge = sorted_edges.pop()
            if not self.edge_matrix[edge] is None:
                continue
            if self.edge_connection[edge, 0] % 3 == 0:
                it = self.edge_connection[edge, 0] // 3
            else:
                it = self.edge_connection[edge, 1] // 3
            w1 = self.edge_matrix[self.node_connection[it, 1]]
            ls1 = self.logscale_edge_matrix[self.node_connection[it, 1]]
            w2 = self.edge_matrix[self.node_connection[it, 2]]
            ls2 = self.logscale_edge_matrix[self.node_connection[it, 2]]
            new_w = oe.contract(
                "abc, ade, fbd, gce -> afg", w1, w2, self.tensor[it], self.tensor[it]
            )
            (new_s, indices) = torch.max(
                torch.reshape(torch.abs(new_w), (nsample, -1)), 1
            )
            new_ls = torch.log(new_s)
            new_w *= torch.unsqueeze(torch.unsqueeze(torch.exp(-new_ls), 1), 2)
            self.edge_matrix[edge] = new_w
            self.logscale_edge_matrix[edge] = new_ls + ls1 + ls2

        return

    def sampling_do(self, i, j, single_site_marginal_distribution=False):
        it = self.edge_connection[self.center_edge, i] // 3
        ie = self.node_connection[it, j]
        itx = self.edge_connection[self.center_edge, 1 - i] // 3
        new_edge_matrix = oe.contract(
            "abc, ade, fbd, gce, f, g -> afg",
            self.edge_matrix[self.node_connection[itx, 1]],
            self.edge_matrix[self.node_connection[itx, 2]],
            self.tensor[itx],
            self.tensor[itx],
            self.center_weight,
            self.center_weight,
        )
        if j == 1:
            new_weight = oe.contract(
                "abc, ade, bfd, cfe -> af",
                new_edge_matrix,
                self.edge_matrix[self.node_connection[it, 2]],
                self.tensor[it],
                self.tensor[it],
            )
        else:
            new_weight = oe.contract(
                "abc, ade, bdf, cef -> af",
                new_edge_matrix,
                self.edge_matrix[self.node_connection[it, 1]],
                self.tensor[it],
                self.tensor[it],
            )
        prob = new_weight[:, 1] / (new_weight[:, 0] + new_weight[:, 1])
        nsample = prob.shape[0]
        new_value = torch.empty(nsample, dtype=torch.uint8, device=self.base_device)
        prob[prob < 0] = 0.0
        prob[prob > 1] = 1.0
        if single_site_marginal_distribution:
            new_edge_matrix = torch.zeros((nsample, 2, 2), device=self.base_device)
            new_edge_matrix[:, 0, 0] = 1
            new_edge_matrix[:, 1, 1] = 1
            self.edge_matrix[ie] = new_edge_matrix
            self.probability_one[:, ie] = prob
        else:
            torch.bernoulli(prob, generator=self.rng, out=new_value)
            self.sampling_data[:, ie] = new_value
            new_edge_matrix = torch.zeros(nsample * 2 * 2, device=self.base_device)
            indices = (
                torch.arange(0, nsample * 2 * 2, 2 * 2, device=self.base_device)
                + new_value * 3
            )
            new_edge_matrix[indices] = 1.0
            self.edge_matrix[ie] = torch.reshape(new_edge_matrix, (nsample, 2, 2))
        self.nsampled += 1
        self.sampled[ie] = True
        return

    def sampling(
        self,
        nsample,
        data=None,
        seed=None,
        single_site_marginal_distribution=False,
        reduced_leaves=None,
    ):
        """
        Variables:
            self.edge_matrix - edge_matrix[i] is torch.tensor(id_sample, k_up, k_down) at edge i
            self.logscale_edge_matrix - logscale_edge_matrix[i] is the logscale of edge_matrix[i][id_sample, k_up, k_down] at edge i
            self.sampled - sampled[i] is True if the leaf i is sampled
            self.nsampled - number of sampled leaves
            self.sampling_data - sampling_data[a, i] is the value of a sampling data at leaf i, a is sample number.
            self.probability_one - probability_one[i] is the probability of 1 at leaf i if single_site_marginal_distribution is True

        Arguments:
            nsample: number of samples
            data: list of (position, status), status is 0 or 1 or status is a vector with nsample 0 or 1 if single_site_marginal_distribution is True
            seed: random seed
            single_site_marginal_distribution: if True, the single-site marginal distribution is calculated
        """
        if not seed is None:
            self.rng.manual_seed(seed)
        self.visit_time.fill(0)
        self.sampling_set_data(
            nsample, data, single_site_marginal_distribution, reduced_leaves
        )
        if single_site_marginal_distribution:
            self.probability_one = torch.zeros(
                (nsample, self.nleaf), device=self.base_device
            )
            for pos, status in data:
                self.probability_one[status == 1, pos] = 1
        current_time = 1
        while self.nsampled < self.nleaf:
            # sampling
            self.visit_time[self.center_edge] = current_time
            current_time += 1
            for i in range(2):
                it = self.edge_connection[self.center_edge, i] // 3
                for j in range(1, 3):
                    ie = self.node_connection[it, j]
                    if self.edge_connection[ie, 0] == -1 and not self.sampled[ie]:
                        self.sampling_do(i, j, single_site_marginal_distribution)
                        self.visit_time[ie] = current_time
                        current_time += 1
            # move
            checked = False
            for i in range(2):
                it = self.edge_connection[self.center_edge, i] // 3
                for j in range(1, 3):
                    ie = self.node_connection[it, j]
                    if self.edge_connection[ie, 0] == -1:
                        continue
                    if checked is False:
                        uncheck_time = self.visit_time[ie]
                        next_pair = (i, j)
                        checked = True
                    else:
                        if uncheck_time > self.visit_time[ie]:
                            next_pair = (i, j)
                            uncheck_time = self.visit_time[ie]
            it = self.edge_connection[self.center_edge, 1 - next_pair[0]] // 3
            new_w = oe.contract(
                "abc, ade, fbd, gce -> afg",
                self.edge_matrix[self.node_connection[it, 1]],
                self.edge_matrix[self.node_connection[it, 2]],
                self.tensor[it],
                self.tensor[it],
            )
            (new_s, indices) = torch.max(
                torch.reshape(torch.abs(new_w), (nsample, -1)), 1
            )
            new_ls = torch.log(new_s)
            new_w *= torch.unsqueeze(torch.unsqueeze(torch.exp(-new_ls), 1), 2)
            self.edge_matrix[self.center_edge] = new_w
            self.logscale_edge_matrix[self.center_edge] = (
                self.logscale_edge_matrix[self.node_connection[it, 1]]
                + self.logscale_edge_matrix[self.node_connection[it, 2]]
                + new_ls
            )
            prev_center = self.center_edge
            self.move_center(next_pair[0], next_pair[1])
        new_w = []
        new_logscale = 0
        for i in range(2):
            it = self.edge_connection[self.center_edge, i] // 3
            new_w.append(
                oe.contract(
                    "abc, ade, fbd, gce -> afg",
                    self.edge_matrix[self.node_connection[it, 1]],
                    self.edge_matrix[self.node_connection[it, 2]],
                    self.tensor[it],
                    self.tensor[it],
                )
            )
            new_logscale += (
                self.logscale_edge_matrix[self.node_connection[it, 1]]
                + self.logscale_edge_matrix[self.node_connection[it, 2]]
            )
        prob0 = oe.contract(
            "abc, abc, b, c -> a",
            new_w[0],
            new_w[1],
            self.center_weight,
            self.center_weight,
        )
        z = torch.sum(self.center_weight * self.center_weight)
        log_p = new_logscale + torch.log(prob0 / z)
        return log_p
