import os


class BaseLogger:
    MAX_VALUE = 0

    def __init__(self,
                 cmd_cfg,
                 column_width=12
                 ):
        self.log_file_loc = cmd_cfg.log_file_loc
        self.save_dir = cmd_cfg.save_dir
        self.model_file_name = cmd_cfg.model_file_name
        self.column_width = column_width
        if os.path.isfile(self.log_file_loc):
            self.logger = open(self.log_file_loc, 'a')
        else:
            self.logger = open(self.log_file_loc, 'w')

    def close_logger(self):
        self.logger.close()

    def write_parameters(self, total_params, total_params_to_update):
        self.logger.write("Total network parameters: %.2f" % total_params)
        self.logger.write("\nTotal parameters to update: %.2f" % total_params_to_update)
        self.logger.flush()

    def save_checkpoint(self, epoch, model, optimizer, loss_tr, score_tr, lr):
        pass

    def save_model(self, epoch, model, score_val):
        pass

    def write_header(self):
        pass

    def write_val(self, epoch, loss_tr, score_tr, score_val):
        pass

    def write_test(self, score_test):
        pass
