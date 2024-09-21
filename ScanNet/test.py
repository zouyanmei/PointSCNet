import torch
from torch import nn
import torch.nn.functional as F
from scannetv2 import ScanNetV2, scan_test_collate_fn
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delasemseg import DelaSemSeg
from config import scan_args, dela_args
from torch.cuda.amp import autocast

torch.set_float32_matmul_precision("high")

# loop x rotation x scaling
#loop = 4 * 4 * 3
loop = 1
cur_id1 = "030102"
os.makedirs(f"output/log/{cur_id1}", exist_ok=True)
logfile = f"output/log/{cur_id1}/test.log"
logfile = open(logfile, "a", 1)
sys.stdout = logfile
testdlr = DataLoader(ScanNetV2(scan_args, partition="val", loop=loop, train=False, test=True), batch_size=1,
                      collate_fn=scan_test_collate_fn, pin_memory=True, num_workers=16)

model1 = DelaSemSeg(dela_args).cuda()
util.load_state(f"output/model/{cur_id1}/best.pt", model=model1)
model1.eval()

metric = util.Metric(20)
cum = 0
cnt = 0

with torch.no_grad():
    for xyz, feature, indices, nn, y, npoint in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast(False):
                p1 = model1(xyz, feature, indices, None, [npoint])

            cum = cum + p1[nn]
            cnt += 1
            if cnt % loop == 0:
                y = y.cuda(non_blocking=True)
                mask = y != 20
                metric.update(cum[mask], y[mask])
                cnt = cum = 0

metric.print("test: ")
