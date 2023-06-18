from .models import CNNClassifier,save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    """

    import torch

    if torch.cuda.is_available():
       m_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
       m_device = torch.device('mps')
    else:
       m_device = torch.device('cpu')

    model = model.to(m_device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    global_step = 0
    for epo in range(args.num_epoch):
        model.train()
        acc_values = []
        for img, label in train_data:
            img, label = img.to(m_device), label.to(m_device)

            logit = model(img)
            loss_value = loss(logit, label)
            acc_value = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_value, global_step)
            acc_values.append(acc_value.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            global_step += 1
        acc_mean = sum(acc_values) / len(acc_values)

        if train_logger:
            train_logger.add_scalar('accuracy', acc_mean, global_step)

        model.eval()
        acc_values = []
        for img, label in valid_data:
            img, label = img.to(m_device), label.to(m_device)
            acc_values.append(accuracy(model(img), label).detach().cpu().numpy())
        vacc_mean = sum(acc_values) / len(acc_values)

        if valid_logger:
            valid_logger.add_scalar('accuracy', vacc_mean, global_step)

        if valid_logger is None or train_logger is None:
            print('epo %-3d \t acc = %0.3f \t val acc = %0.3f' % (epo, acc_mean, vacc_mean))
        save_model(model)
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train(args)
