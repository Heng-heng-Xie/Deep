from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data


def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    import torch
    from os import path
    if torch.cuda.is_available():
        m_device = torch.device('cuda')
    else:
        m_device = torch.device('cpu')

    model.to(m_device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))
    # loss and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.8)
    loss = ClassificationLoss()

    t_data = load_data('data/train')
    v_data = load_data('data/valid')

    for epo in range(args.num_epoch):
        model.train()
        loss_value = []
        acc_value = []
        vacc_value = []
        for im, l in t_data:
            im, l = im.to(m_device), l.to(m_device)
            logit = model(im)
            loss_value = loss(logit, l)
            acc_value = accuracy(logit, l)
            # loss and acc value.
            loss_value.append(loss_value.detach().cpu().numpy())
            acc_value.append(acc_value.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        loss_mean = sum(loss_value) / len(loss_value)
        acc_mean = sum(acc_value) / len(acc_value)

        model.eval()
        for im, l in v_data:
            im, l = im.to(m_device), l.to(m_device)
            vacc_value.append(accuracy(model(im), l).detach().cpu().numpy())
        vacc_mean = sum(vacc_value) / len(vacc_value)

        print('epo %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epo, loss_mean, acc_mean, vacc_mean))
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
