import torch
from libs.utils.logger.base_logger import BaseLogger


class BCDLogger(BaseLogger):
    def __init__(self,
                 cmd_cfg,
                 column_width=12
                 ):
        super(BCDLogger, self).__init__(cmd_cfg=cmd_cfg, column_width=column_width)

    def save_checkpoint(self, epoch, model, optimizer, loss_tr, score_tr, lr):
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_tr': loss_tr,
            'score_tr': score_tr['F1'],
            'lr': lr
        }, self.save_dir + 'checkpoint.pth.tar')

    def save_model(self, epoch, model, score_val):
        if epoch % 1 == 0 and self.MAX_VALUE <= score_val['F1']:
            self.MAX_VALUE = score_val['F1']
            torch.save(model.state_dict(), self.model_file_name)

    def write_header(self):
        header = "\n{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}".format(
            'Epoch', 'Kappa', 'IoU', 'F1', 'Rec', 'Pre', width=self.column_width
        )
        self.logger.write(header)
        self.logger.flush()

    def write_val(self, epoch, loss_tr, score_tr, score_val):
        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.2f\t Score(tr) = %.2f\t Score(val) = %.2f"
              % (epoch, loss_tr, score_tr['F1'], score_val['F1']))

        val_line = ("\n{: ^{width}}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}"
                    "|{: ^{width}.2f}").format(
            epoch, score_val['Kappa'], score_val['IoU'], score_val['F1'],
            score_val['Rec'], score_val['Pre'], width=self.column_width
        )
        self.logger.write(val_line)
        self.logger.flush()

    def write_test(self, score_test):
        print("\nTest :\t Kappa (te) = %.2f\t IoU (te) = %.2f\t F1 (te) = %.2f"
              % (score_test['Kappa'], score_test['IoU'], score_test['F1']))
        test_line = (
            "\n{: ^{width}}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}").format(
            'Test', score_test['Kappa'], score_test['IoU'], score_test['F1'],
            score_test['Rec'], score_test['Pre'], width=self.column_width
        )
        self.logger.write(test_line)
        self.logger.flush()
