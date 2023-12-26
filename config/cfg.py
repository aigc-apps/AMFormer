import argparse
import yaml
import os
from datetime import datetime
import time

def arg2str(args):
    # args.run_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args_dict = vars(args)
    option_str = 'run_time: ' + datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str


class BaseConfig(object):

    def __init__(self, config = None):
        self.config = config

        self.parser = argparse.ArgumentParser()
        # self.parser.add_argument('--confag', type=str, default='a')
        self.parser.add_argument('--config', type=str, default='/home/ceyu.cy/projects/tabular/Tablular_ours/config/configs/base.yaml')
        # print(self.parser.config)
        # print(self.parser.confag)
        # raise ValueError

    
    def load_base(self, derived_config, config):
        if '__base__' in derived_config:
            for each in derived_config['__base__']:
                with open(each) as f:
                    derived_config_ = yaml.safe_load(f)
                    config = self.load_base(derived_config_, config)
            # config = {**config, **derived_config}
        # else:
        config = {**config, **derived_config}
        return config

    # def load_base(self, derived_config, config):
    #     config = self._load_base(derived_config, config)
    #     if config['exp_param'] not in [None, 'None']:
    #         for each in config['exp_param']:
    #             config['exp_name'] = config['exp_name'] + '-' + each + '=' + config[each]

    #     return config

    def initialize(self, config = None):
        args = self.parser.parse_args()

        # print(self.parser.config)
        # print(self.parser.confag)
        # raise ValueError

        if self.config:
            args.config = self.config

        config = {}
        with open(args.config) as f:
            derived_config = yaml.safe_load(f)
            config = self.load_base(derived_config, config)



        if 'exp_param' in config and config['exp_param'] not in [None, 'None']:
            if isinstance(config['exp_param'], str):
                config['exp_name'] = str(config['exp_name']) + '-' + str(config['exp_param']) + '=' + str(config[config['exp_param']])
            else:
                for each in config['exp_param']:
                    config['exp_name'] = str(config['exp_name']) + '-' + str(each) + '=' + str(config[each])


                
        for key, value in config.items():
            setattr(args, key, value)


        if args.time_delay != 0:
            print('================ {:^30s} ================'.format('Delay for {} seconds'.format(args.time_delay)))
            time.sleep(args.time_delay)

        return args


    def save_result(self, acc_list, loss_list, loss_refine_list):

        acc_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/acc_{}_std_{}.txt'.format(acc_mean, acc_std)
        loss_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss_{}_std_{}.txt'.format(loss_mean, loss_std)
        loss_refine_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss2_{}_std_{}.txt'.format(loss_refine_mean, loss_refine_std)

        with open(acc_save_path, "w") as f:
            for c, ac in enumerate(acc_list):
                f.write('train_{}_acc is {}\n'.format(c, ac))
        f.close()

        with open(loss_save_path, "w") as f:
            for c, ac in enumerate(loss_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()

        with open(loss_refine_save_path, "w") as f:
            for c, ac in enumerate(loss_refine_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()


# 加载YAML文件
# if args.config:
#     with open(args.config) as f:
#         config = yaml.safe_load(f)
# else:
#     # 如果未指定YAML文件，则使用默认值
#     config = {
#         'arg1': 'default_value1',
#         'arg2': 123,
#         'arg3': 3.14
#     }

# # 将YAML配置加载到args中
# for key, value in config.items():
#     setattr(args, key, value)

# # 输出参数值
# print(args.arg1)
# print(args.arg2)
# print(args.arg3)
