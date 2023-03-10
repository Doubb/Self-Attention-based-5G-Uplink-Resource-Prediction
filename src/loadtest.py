import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle5 as pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer


def main(config):
    #os.environ["NUMEXPR_MAX_THREADS"] = "96"
    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'],
                         limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Split dataset
    # test_data = my_data
    test_indices = None

    if config['test_pattern']:  # used if test data come from different files / file patterns
        test_data = my_data
        test_indices = test_data.all_IDs

    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    #elif config['normalization'] is not None:
    #    normalizer = Normalizer(config['normalization'])
    #    my_data.feature_df.loc[test_indices] = normalizer.normalize(my_data.feature_df.loc[test_indices])
    #    if not config['normalization'].startswith('per_sample'):
            # get normalizing values from training set and store for future use
    #        norm_dict = normalizer.__dict__
    #        with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
    #            pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        #if len(val_indices):
        #    val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
        if len(test_indices):
            norm_start_time = time.time()
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])
            norm_run_time = time.time() - norm_start_time
            logger.info("Normalization runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(norm_run_time)))

    print('==========Print Test Data=============')
    print(test_data.feature_df)
    print('==========Print Test Data=============')
    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, my_data)
    #model = torch.load(config['load_model'])
    model.load_state_dict(torch.load(config['load_model'])['state_dict'])

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step

    # Load model and optimizer state
    if args.load_model:
        logger.info("Loading model ...")
        #model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
        #                                                 config['change_output'],
        #                                                 config['lr'],
        #                                                 config['lr_step'],
        #                                                 config['lr_factor'])
        #model = torch.load(config['load_model'])
        logger.info("Loaded model from {}".format(config['load_model']))
    model.to(device)
    model.eval()

    loss_module = get_loss_module(config)

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                      print_interval=config['print_interval'], console=config['console'])
        #aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True, testing=True)
        best_metrics = {}
        best_value = 100000
        tensorboard_writer = SummaryWriter(config['tensorboard_dir'])
        aggr_metrics_test, best_metrics, best_value = validate(test_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch=0)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        return


if __name__ == '__main__':
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
