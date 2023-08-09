import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data

import torch.utils.tensorboard as tb
import torchvision


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    if torch.cuda.is_available():
       m_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
       m_device = torch.device('mps')
    else:
       m_device = torch.device('cpu')


    model = model.to(m_device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_of_det = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_of_size = torch.nn.MSELoss(reduction='none')

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)



    transform =  torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor()])


    train_data = load_detection_data('training_data/train', num_workers=4, batch_size=60, transform=transform)


    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        for img, gt_det, gt_size in train_data:
            img, gt_det, gt_size = img.to(m_device), gt_det.to(m_device), gt_size.to(m_device)
            size_w, _ = gt_det.max(dim=1, keepdim=True)
            det, size = model(img)

            # weight size loss and det loss
            pre_det = torch.sigmoid(det * (1-2*gt_det))
            det_loss_val = (loss_of_det(det, gt_det)*pre_det).mean() / pre_det.mean()
            size_loss_val = (size_w * loss_of_size(size, gt_size)).mean() / size_w.mean()
            loss_val = det_loss_val + size_loss_val * args.size_weight

            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, gt_det, det, global_step)

            if train_logger is not None:
                train_logger.add_scalar('det_loss', det_loss_val, global_step)
                train_logger.add_scalar('size_loss', size_loss_val, global_step)
                train_logger.add_scalar('loss', loss_val, global_step)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t det_loss = %0.3f \t size_loss = %0.3f \t loss = %0.3f' %
                  (epoch, det_loss_val, size_loss_val, loss_val))

        #scheduler.step()

    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    train(args)
