import os
import wandb
from tqdm import tqdm
import torch
import logging
import argparse
from vis_transformer import viT
import torch.optim as optim
from scheduler import CosineLRScheduler
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast
from utils import NativeScaler, AverageMeter, update_summary
from dataset import create_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    wandb.init(project=args.name, config=args)
    # model
    model = viT(
        img_size=args.img_size,
        patch_size=16,
        num_class=args.num_class,
        embed_dim=192,
        num_encoder=args.num_layer,
        num_head=3,
        mlp_dim=768,
        dropout_rate=0.1,
        global_pool=args.global_pool,
        use_conv=args.use_conv
    ).cuda()
    
    '''
    model.load_state_dict(torch.load('./StanfordCars_6_False_True__True_199.pt'))
    '''

    # count parameter
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: %.4fM" % (params/1000000))

    # optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(optimizer, args.warmup_steps, args.num_epoch)

    # amp
    scaler = NativeScaler()

    # create data loader (cutmix...?nah randomaug?in torch)
    train_loader, test_loader = create_dataloader(args, args.dataset, args.randaug)

    # loss function
    loss_func = CrossEntropyLoss()

    best_acc = 0

    for epoch in tqdm(range(args.num_epoch)):
        metric_train = train(model, train_loader,
                             loss_func, optimizer, scheduler, scaler)

        metric_val = validate(model, test_loader, loss_func)

        logger.info("train loss: %.4f   val loss: %.4f   val accuracy: %.4f",
                    metric_train['loss'], metric_val['loss'], metric_val['acc'])

        if metric_val['acc'] > best_acc:
            best_acc = metric_val['acc']

        update_summary(epoch, metric_train, metric_val)

        # save model
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), f'./{args.dataset}_{args.num_layer}_{args.global_pool}_{args.randaug}__{args.use_conv}_{epoch}.pt')

    logger.info('Best acc: %.2f' % best_acc)


def train(model, train_loader, loss_func, optimizer, scheduler, scaler):
    model.train()
    losses = AverageMeter()

    for batch_idx, (X, target) in enumerate(train_loader):
        X, target = X.cuda(), target.cuda()

        with autocast():
            pred = model(X)
            loss = loss_func(pred, target)

        losses.update(loss.item(), X.size(0))

        # optimize
        optimizer.zero_grad()
        scaler(loss, optimizer)
        scheduler.step()

    return {"loss": losses.avg}


def validate(model, test_loader, loss_func):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for batch_idx, (X, target) in enumerate(test_loader):
            X, target = X.cuda(), target.cuda()

            with autocast():
                output = model(X)

            loss = loss_func(output, target)

            _, pred = output.max(dim=1)
            acc = (pred == target).sum()/output.size(0)
            losses.update(loss.item(), X.size(0))
            accs.update(acc.item(), output.size(0))

    return {"loss": losses.avg, "acc": accs.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--num_epoch", default=200, type=int)
    parser.add_argument("--num_class", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--dataset", default='cifar10', type=str)
    parser.add_argument("--name", default='cifar10', type=str)
    parser.add_argument("--num_layer", default='6', type=int)
    parser.add_argument("--global_pool", default=False, type=bool)
    parser.add_argument("--randaug", default=False, type=bool)
    parser.add_argument("--use_conv", default=False, type=bool)

    args = parser.parse_args()

    main(args)
