import random
import torch
import numpy as np
def wrap_str(*args):
    return ' '.join([str(a) for a in args])


def safe_state(seed=0, silent=None):
    # old_f = sys.stdout
    # class F:
    #     def __init__(self, silent):
    #         self.silent = silent
    #
    #     def write(self, x):
    #         if not self.silent:
    #             if x.endswith("\n"):
    #                 old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
    #             else:
    #                 old_f.write(x)
    #
    #     def flush(self):
    #         old_f.flush()
    #
    # sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))