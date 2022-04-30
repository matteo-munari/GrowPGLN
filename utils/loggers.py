import os


class Logger:
    def __init__(self, config):
        self.config = config
        self.logs = []
        self.ignore = {'log_dir', 'exp_seed'}

    def append(self, log):
        self.logs.append(log)

    def print_logs(self):
        for key in self.config:
            print(f'{key.title()}: {self.config[key]}')
        for log in self.logs:
            print(log)

    def save_to_file(self):
        exp_filepath = self.config['log_dir'] + '/experiment_setup.txt'
        if not os.path.exists(exp_filepath):
            with open(exp_filepath, 'w') as f:
                f.write('EXPERIMENT SETUP\n\n')
                _ = [f.write(f'{key.title()}: {self.config[key]}\n') for key in self.config if key not in self.ignore]

        log_filepath = self.config['log_dir'] + f'/log_seed_{self.config["exp_seed"]}.txt'
        with open(log_filepath, 'w') as f:
            f.write('LOG\n\n')
            _ = [f.write(log+'\n') for log in self.logs]
