import math


def next_batch(X1, X2, batch_size):
    # generate next batch
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size) - 1  # fix the last batch
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, (i + 1))


def next_batch_gt(X1, X2, gt, batch_size):
    # generate next batch with label
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size) - 1  # fix the last batch
    for i in range(int(total) - 1):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        gt_now = gt[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, gt_now, (i + 1))


def next_batch_3view(X1, X2, X3, batch_size):
    # generate next batch for 3 view data
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)  # fix the last batch
    for i in range(int(total) - 1):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, (i + 1))


def next_batch_gt_3view(X1, X2, X3, gt, batch_size):
    # generate next batch for 3 view data with label
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)  # fix the last batch
    for i in range(int(total) - 1):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        gt_now = gt[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, gt_now, (i + 1))
