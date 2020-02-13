import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from hssm import EnvModel
from utils import preprocess, postprocess, full_dataloader, log_train, log_test, plot_rec, plot_gen
LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="vta agr parser")
    parser.add_argument('--seed', type=int, default=111)

    # data size
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-size', type=int, default=20)
    parser.add_argument('--init-size', type=int, default=5)

    # model size
    parser.add_argument('--state-size', type=int, default=8)
    parser.add_argument('--belief-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)

    # observation distribution
    parser.add_argument('--obs-std', type=float, default=1.0)
    parser.add_argument('--obs-bit', type=int, default=5)

    # optimization
    parser.add_argument('--learn-rate', type=float, default=0.0005)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--max-iters', type=int, default=500000)

    # subsequence prior params
    parser.add_argument('--seg-num', type=int, default=5)
    parser.add_argument('--seg-len', type=int, default=8)

    # gumbel params
    parser.add_argument('--max-beta', type=float, default=1.0)
    parser.add_argument('--min-beta', type=float, default=0.1)
    parser.add_argument('--beta-anneal', type=float, default=100)

    # log dir
    parser.add_argument('--log-dir', type=str, default='./asset/log/')
    return parser.parse_args()


def set_exp_name(args):
    exp_name = 'hssm_maze'
    exp_name += '_b{}'.format(args.batch_size)
    exp_name += '_l{}_i{}'.format(args.seq_size, args.init_size)
    exp_name += '_b{}_s{}_c{}'.format(args.belief_size, args.state_size, args.num_layers)
    exp_name += '_gc{}_lr{}'.format(args.grad_clip, args.learn_rate)
    exp_name += '_sg{}-{}'.format(args.seg_num, args.seg_len)
    exp_name += '_std{}_bits{}'.format(args.obs_std, args.obs_bit)
    exp_name += '_gum{}-{}-{}'.format(args.min_beta, args.max_beta, args.beta_anneal)
    exp_name += '_seed{}'.format(args.seed)
    return exp_name


def main():
    # parse arguments
    args = parse_args()

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # set logger
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

    # set size
    seq_size = args.seq_size
    init_size = args.init_size

    # set device as gpu
    device = torch.device('cuda', 0)

    # set writer
    exp_name = set_exp_name(args)
    writer = SummaryWriter(args.log_dir + exp_name)
    LOGGER.info('EXP NAME: ' + exp_name)

    # load dataset
    train_loader, test_loader = full_dataloader(seq_size, init_size, args.batch_size)
    LOGGER.info('Dataset loaded')

    # init models
    model = EnvModel(belief_size=args.belief_size,
                     state_size=args.state_size,
                     num_layers=args.num_layers,
                     max_seg_len=args.seg_len,
                     max_seg_num=args.seg_num).to(device)
    LOGGER.info('Model initialized')

    # init optimizer
    optimizer = Adam(params=model.parameters(),
                     lr=args.learn_rate, amsgrad=True)

    # test data
    pre_test_full_data_list = iter(test_loader).next()
    pre_test_full_data_list = preprocess(pre_test_full_data_list.to(device), args.obs_bit)

    # for each iter
    b_idx = 0
    while b_idx <= args.max_iters:
        # for each batch
        for train_obs_list in train_loader:
            b_idx += 1
            # mask temp annealing
            if args.beta_anneal:
                model.state_model.mask_beta = (args.max_beta - args.min_beta) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
            else:
                model.state_model.mask_beta = args.max_beta

            ##############
            # train time #
            ##############
            # get input data
            train_obs_list = preprocess(train_obs_list.to(device), args.obs_bit)

            # run model with train mode
            model.train()
            optimizer.zero_grad()
            results = model(train_obs_list, seq_size, init_size, args.obs_std)

            # get train loss and backward update
            train_total_loss = results['train_loss']
            train_total_loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # log
            if b_idx % 10 == 0:
                log_str, log_data = log_train(results, writer, b_idx)
                LOGGER.info(log_str, *log_data)

            #############
            # test time #
            #############
            if b_idx % 100 == 0:
                # set data
                pre_test_init_data_list = pre_test_full_data_list[:, :init_size]
                post_test_init_data_list = postprocess(pre_test_init_data_list, args.obs_bit)
                pre_test_input_data_list = pre_test_full_data_list[:, init_size:(init_size + seq_size)]
                post_test_input_data_list = postprocess(pre_test_input_data_list, args.obs_bit)

                with torch.no_grad():
                    ##################
                    # test data elbo #
                    ##################
                    model.eval()
                    results = model(pre_test_full_data_list, seq_size, init_size, args.obs_std)
                    post_test_rec_data_list = postprocess(results['rec_data'], args.obs_bit)
                    output_img, output_mask = plot_rec(post_test_init_data_list,
                                                       post_test_input_data_list,
                                                       post_test_rec_data_list,
                                                       results['mask_data'],
                                                       results['p_mask'],
                                                       results['q_mask'])

                    # log
                    log_str, log_data = log_test(results, writer, b_idx)
                    LOGGER.info(log_str, *log_data)
                    writer.add_image('valid/rec_image', output_img.transpose([2, 0, 1]), global_step=b_idx)
                    writer.add_image('valid/mask_image', output_mask.transpose([2, 0, 1]), global_step=b_idx)

                    ###################
                    # full generation #
                    ###################
                    pre_test_gen_data_list, test_mask_data_list = model.full_generation(pre_test_init_data_list, seq_size)
                    post_test_gen_data_list = postprocess(pre_test_gen_data_list, args.obs_bit)

                    # log
                    output_img = plot_gen(post_test_init_data_list, post_test_gen_data_list, test_mask_data_list)
                    writer.add_image('valid/full_gen_image', output_img.transpose([2, 0, 1]), b_idx)


if __name__ == '__main__':
    main()
