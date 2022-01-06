import numpy
import sys
import time
from tqdm import tqdm
import torch
import torchvision

from xview3.postprocess.v2.model_simple import Model
from xview3.postprocess.v2.dataset import Dataset

model_path = sys.argv[1]
csv_path = sys.argv[2]
image_path = sys.argv[3]

batch_size = 32
num_loader_workers = 4

train_dataset = Dataset(filter_func=lambda idx: idx%10 != 0, csv_path=csv_path, image_path=image_path)
val_dataset = Dataset(filter_func=lambda idx: idx%10 == 0, csv_path=csv_path, image_path=image_path)

train_sampler = torch.utils.data.RandomSampler(train_dataset)
val_sampler = torch.utils.data.SequentialSampler(val_dataset)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=num_loader_workers,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=val_sampler,
    num_workers=num_loader_workers,
)

model = Model()

device = torch.device("cuda")
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)
best_accuracy = 0

for epoch in range(200):
    print('begin epoch {}'.format(epoch))
    t0 = time.time()

    model.train()
    optimizer.zero_grad()
    train_losses = {}

    for images, targets, _ in tqdm(train_loader):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        loss_dict = model(images, targets=targets)
        loss = loss_dict['loss']

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for k, v in loss_dict.items():
            if k not in train_losses:
                train_losses[k] = []
            train_losses[k].append(v.item())

    model.eval()
    val_accs = {}
    with torch.no_grad():
        for images, targets, _ in tqdm(val_loader):
            images = images.to(device)
            t = model(images)
            t = [x.cpu() for x in t]
            pred_length, pred_confidence, pred_correct, pred_source, pred_fishing, pred_vessel = t

            def get_cls_acc(pred, target):
                valid_indices = target >= 0
                valid_pred = pred[valid_indices, :]
                valid_target = target[valid_indices]
                if len(valid_target) == 0:
                    return (1.0, 0)

                valid_pred_cls = valid_pred.argmax(dim=1)
                matches = (valid_pred_cls == valid_target).sum().float()
                return (float(matches / len(valid_target)), int(len(valid_target)))

            valid_indices = targets['vessel_length'] > 0
            valid_pred = pred_length[valid_indices]
            valid_target = targets['vessel_length'][valid_indices]
            if len(valid_target) > 0:
                length_acc = 1 - torch.div(torch.abs(valid_pred - valid_target), valid_target).mean().item()
                length_weight = len(valid_target)
            else:
                length_acc = 1.0
                length_weight = 0

            accs = {
                'length': (length_acc, length_weight),
                'confidence': get_cls_acc(pred_confidence, targets['confidence']),
                'correct': get_cls_acc(pred_correct, targets['correct']),
                'source': get_cls_acc(pred_source, targets['source']),
                'fishing': get_cls_acc(pred_fishing, targets['fishing']),
                'vessel': get_cls_acc(pred_vessel, targets['vessel']),
            }
            for k, v in accs.items():
                if k not in val_accs:
                    val_accs[k] = []
                val_accs[k].append(v)

    train_losses = {k: numpy.mean(v) for k, v in train_losses.items()}
    train_loss = train_losses['loss']
    val_accs = {
        k:
        numpy.average(
            [x[0] for x in v],
            weights=[x[1] for x in v]
        )
        for k, v in val_accs.items()
    }
    val_acc = numpy.mean(list(val_accs.values()))

    print('epoch {}: train_loss={} val_acc={} elapsed={} lr={}'.format(
        epoch,
        train_loss,
        val_acc,
        int(time.time()-t0),
        optimizer.param_groups[0]['lr'],
    ))
    print('train_losses:', train_losses)
    print('val_accs:', val_accs)

    lr_scheduler.step(train_loss)

    if val_acc > best_accuracy:
        print('save model', val_acc, best_accuracy)
        best_accuracy = val_acc
        torch.save(model.state_dict(), model_path)
