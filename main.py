"""
MFNet: Multi-filter Directive Network for Weakly Supervised Salient Object Detection
Conference: 2021 IEEE/CVF International Conference on Computer Vision, poster
Author: JianWang (Yimmy)
Contact: jiangnanyimi@163.com  or  dlyimi@mail.dlut.edu.cn
College: The IIAU-OIP Lab, Dalian University of Technology
"""


def main():
    import os
    import time
    import torch
    import argparse

    from trainsal import TrainSal
    # from utils.crf import crf
    from utils.imsave import imsave
    from utils.pamr import BinaryPamr

    from utils.datainit import traindatainit
    from model.MFNet_densenet import MFNet
    from torch.utils.data import DataLoader
    from dataset_loader import MySalTrainData, MySalInferData, MySalValData

    # -------------------------------------------------- options --------------------------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=str, default='train', choices=['train', 'infer'])
    parser.add_argument('--num_workers', type=int, default=12, help='the CPU workers number')
    parser.add_argument('--resize', type=int, default=256, help='resized size of images')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='wight_decay')
    parser.add_argument('--ckpt_root', type=str, default='snapshot', help='path to save ckpt')

    parser.add_argument('--sal_stage', type=int, default=2, help='the iterations of the self-training scheme')
    parser.add_argument('--lr', type=float, default=3e-6, help='learning rate')
    parser.add_argument('--batch', type=int, default=25, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=10, help='the max epoch')
    parser.add_argument('--k', type=int, default=30, help='the extra epoch of train stage n')
    parser.add_argument('--val', type=bool, default=False, help='whether validation or not')
    parser.add_argument('--data_root', type=str, default='data', help='path to infer and train data')
    args = parser.parse_args()
    print(args)
    ori_root = os.getcwd()
    traindatainit(args.ckpt_root, args.data_root, args.sal_stage)

    # ------------------------------------------------ dataloaders ------------------------------------------------- #
    infersal_loader = DataLoader(MySalInferData(args.data_root, transform=True), batch_size=args.batch,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # -------------------------------------------------- networks -------------------------------------------------- #
    model_sal = MFNet()
    model_sal = model_sal.cuda()

    # -------------------------------------------------- training -------------------------------------------------- #

    print('\n[ Training a saliency network using pseudo labels. ]\n')
    for i in range(1, (args.sal_stage*2)):

        os.chdir(ori_root)
        # train
        if args.param == 'train':
            model_sal.train()
            args.param = 'infer'

            trainsal_loader = DataLoader(MySalTrainData(args.data_root, resize=args.resize, transform=True,
                                                        stage=int((i-1)/2)), batch_size=args.batch,
                                         shuffle=True, num_workers=args.num_workers, pin_memory=True)
            optimizer_model = torch.optim.Adam(model_sal.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            if not args.val:
                valsal_loader = None
            else:
                valsal_loader = DataLoader(MySalValData(args.data_root, resize=args.resize, transform=True),
                                           batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            training = TrainSal(
                model=model_sal,
                optimizer_model=optimizer_model,
                train_loader=trainsal_loader,
                val_loader=valsal_loader,
                outpath=args.ckpt_root,
                max_epoch=args.max_epoch + args.k*((i+1)/2-1),
                stage=(i+1)/2)

            training.epoch = 0
            training.iteration = 0
            training.train()

    # --------------------------------------------------- infer ---------------------------------------------------- #
        elif args.param == 'infer':
            args.param = 'train'
            torch.cuda.empty_cache()
            ckpt_name = 'sal_stage_' + str(int(i/2)) + '.pth'
            model_sal.load_state_dict(torch.load(os.path.join(args.ckpt_root, ckpt_name)))
            model_sal.eval()

            print('\nInferring the saliency maps and pixel-wise pseudo labels .....    ', end='')

            total_num = len(infersal_loader)
            count_num = int(total_num / 10)
            start_time = time.time()

            with torch.no_grad():
                for idx, (data, name, size) in enumerate(infersal_loader):
                    _, _, sal = model_sal.forward(data.cuda())

                    # Performing pixel-wise refinements on the generated saliency maps.
                    sal_pamr = BinaryPamr(data.cuda(), sal.detach(), binary=0.4)
                    sal_pamr = sal_pamr.squeeze().cpu().detach()
                    sal = sal.squeeze().cpu().detach()

                    for index in range(sal.shape[0]):  # Saving the maps
                        img_size = [[size[0][index].item()], [size[1][index].item()]]
                        imsave(os.path.join('data/pseudo_labels/label0_' + str(int(i / 2)), name[index] + '.png'),
                               sal_pamr[index], img_size, True)
                        imsave(os.path.join('data/pseudo_labels/label1_' + str(int(i / 2)), name[index] + '.png'),
                               sal[index], img_size, False)

                    if idx % count_num == count_num - 1:
                        print((str(round(int(idx + 1) / total_num * 100))) + '.0 %   ', end='')

            print(',  finished,  ', end='')
            final_time = time.time()
            print('cost %d seconds.  ' % (final_time - start_time), end='\n\n')
            torch.cuda.empty_cache()

            # Performing superpixel-wise refinements as well as CRF on the generated saliency maps.
            from utils.slic import run_slic_with_crf
            print('\nInferring superpixel-wise pseudo labels .....   \n[  ', end='')
            start_time = time.time()
            run_slic_with_crf(img_root='data/DUTS-train/image',
                              prob_root='data/pseudo_labels/label1_' + str(int(i/2)),
                              output_root='data/pseudo_labels/label1_' + str(int(i/2)))
            print(' ],  finished,  ', end='')
            final_time = time.time()
            print('cost %d seconds.  ' % (final_time - start_time), end='\n\n')

            # # ------------- CRF ------------- #
            # print('\nCRFing .....   \n[  ', end='')
            # start_time = time.time()
            #
            # crf(input_path=r'data/DUTS-train/image', sal_path='data/pseudo_labels/label0_' + str(int(i/2)),
            #     output_path='data/pseudo_labels/label0_' + str(int(i/2)), binary=None)
            # os.chdir(ori_root)
            # print(' ],  finished,  ', end='')
            # final_time = time.time()
            # print('cost %d seconds.  ' % (final_time - start_time), end='\n\n')

            # Reload the model for self-training
            del model_sal
            model_sal = MFNet()
            model_sal = model_sal.cuda()


if __name__ == '__main__':
    main()
