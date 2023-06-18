from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop

    for epoch in range(10):
        torch.manual_seed(epoch)
        acc = []

        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_logger.add_scalar('loss', dummy_train_loss, global_step=20*epoch+iteration)
            acc.append(torch.mean(dummy_train_accuracy))
           # expect = epoch / 10. + torch.mean(torch.cat([torch.randn(10) for i in range(20)]))
        train_logger.add_scalar('accuracy', sum(acc)/len(acc), global_step=20*epoch+20)
        torch.manual_seed(epoch)
        val_acc = []
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            val_acc.append(torch.mean(dummy_validation_accuracy))
        valid_logger.add_scalar('accuracy', sum(val_acc)/len(val_acc), global_step=20*epoch+20)





if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
