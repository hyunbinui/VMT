import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

import argparse
import time
import datetime
import logging
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from model import make_model
from utils import set_logger,read_vocab,write_vocab,build_vocab,Tokenizer,clip_gradient,adjust_learning_rate
from dataloader import create_split_loaders
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
cc = SmoothingFunction()
from label_smoothing import *
from utils import padding_idx, sos_idx, eos_idx, unk_idx, NoamOpt, SimpleLossCompute, beam_search_decode, generate_tgt
from tqdm import tqdm


en_input, ko_input = 'en', 'ko'


class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])


def save_checkpoint(state, cp_file):
    torch.save(state, cp_file)


def setup(args, clear=False):
    '''
    Build vocabs from train or train/val set.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_KO = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_KO
    if clear: ## delete previous vocab
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_KO]:
            if os.path.exists(file):
                os.remove(file)
    # Build English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'),  TRAIN_VOCAB_EN)
    #build Korean vocabs
    if not os.path.exists(TRAIN_VOCAB_KO):
        write_vocab(build_vocab(args.DATA_DIR, language='ko'), TRAIN_VOCAB_KO)

    # set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id)

    log_path = args.LOG_DIR + model_prefix + '/'
    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"
    init_epoch = 0

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## set up the logger
    set_logger(os.path.join(log_path, 'train.log'))

    ## save argparse parameters
    with open(log_path + 'args.yaml', 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))

    logging.info('Training model: {}'.format(model_prefix))

    ## set up vocab txt
    setup(args, clear=True)
    print(args.__dict__)

    maps = {'en': args.TRAIN_VOCAB_EN, 'ko': args.TRAIN_VOCAB_KO}
    vocab_en = read_vocab(maps[en_input])
    tok_en = Tokenizer(language=en_input, vocab=vocab_en, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_ko = read_vocab(maps[ko_input])
    tok_ko = Tokenizer(language=ko_input, vocab=vocab_ko, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size en/ko:{}/{}'.format(len(vocab_en), len(vocab_ko)))

    ## Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_en, tok_ko), args.batch_size,
                                                                 args.MAX_VID_LENGTH, (en_input, ko_input),
                                                                 num_workers=4, pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format(len(train_loader), len(val_loader), len(test_loader)))

    ## init model
    model = make_model(len(vocab_en), len(vocab_ko), N=args.nb_blocks, d_model=args.d_model, d_ff=args.d_model * 4, h=args.att_h, dropout=args.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    ## define loss
    criterion_en2ko = LabelSmoothing(size=len(vocab_ko), padding_idx=padding_idx, smoothing=args.smoothing)
    criterion_ko2en = LabelSmoothing(size=len(vocab_en), padding_idx=padding_idx, smoothing=args.smoothing)
    criterion = (criterion_en2ko, criterion_ko2en)

    label_criterion = LabelSmoothing(size=401, padding_idx=padding_idx, smoothing=args.smoothing)

    ## init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-9)
    model_opt = NoamOpt(args.d_model, 1, args.warmup_steps, optimizer)

    ## track loss during training
    total_train_loss, total_val_loss = [], []
    best_val_bleu, best_epoch = 0, 0

    ## init time
    zero_time = time.time()

    # Begin training procedure
    earlystop_flag = True

    if True:

        for epoch in range(init_epoch, args.epochs):
            ## train for one epoch
            start_time = time.time()
            train_loss = train(train_loader, model, SimpleLossCompute(model.en_generator, model.ko_generator, criterion, opt=model_opt), epoch, tok_ko, tok_en, label_criterion)

            val_loss, corpbleu_en2ko, corpbleu_ko2en = validate(val_loader, model, SimpleLossCompute(model.en_generator, model.ko_generator, criterion, opt=None), epoch, label_criterion)
            end_time = time.time()

            epoch_time = end_time - start_time
            total_time = end_time - zero_time

            logging.info('Total time used: %s Epoch %d time used: %s train loss: %.4f val loss: %.4f corpbleu_en2ko: %.4f corpbleu_ko2en: %.4f' % (
                str(datetime.timedelta(seconds=int(total_time))),
                epoch, str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, corpbleu_en2ko, corpbleu_ko2en))

            corpbleu = corpbleu_en2ko + corpbleu_ko2en

            if corpbleu > best_val_bleu:
                best_val_bleu = corpbleu
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(),
                                 'optimizer': model_opt.optimizer.state_dict()}, cp_file)
                best_epoch = epoch

            logging.info("Finished {0} epochs of training".format(epoch + 1))


            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('validation_loss', val_loss, epoch)
            writer.add_scalar('corpbleu_en2ko', corpbleu_en2ko, epoch)
            writer.add_scalar('corpbleu_ko2en', corpbleu_ko2en, epoch)
            writer.add_scalar('corpbleu', corpbleu, epoch)
            writer.add_scalar('best_val_bleu', best_val_bleu, epoch)

            writer.flush()
            writer.close()


            total_train_loss.append(train_loss)
            total_val_loss.append(val_loss)

            if earlystop_flag:
                if epoch - best_epoch >= 12:
                    break

        logging.info('Best corpus bleu score {:.4f} at epoch {}'.format(best_val_bleu, best_epoch))

        ### the best model is the last model saved in our implementation
        logging.info('************ Start eval... ************')
        eval(test_loader, model, cp_file, tok_ko, tok_en, nbest=1, result_path=result_path)


def train(train_loader, model, loss_compute, epoch, tok_ko, tok_en, act_label_loss_compute):
    '''
    Performs one epoch's training.
    '''
    model.train()
    total_tokens = 0
    total_loss = 0
    for (encap, kocap), (ensrccap_mask, kosrccap_mask), (kotgt_mask, entgt_mask), video, video_mask, _, _, _, act_labels in tqdm(train_loader, desc="epoch {}/{}".format(epoch + 1, args.epochs)):
         
        encap, ensrccap_mask, entgt_mask, kocap, kosrccap_mask, kotgt_mask = encap.cuda(), ensrccap_mask.cuda(), entgt_mask.cuda(), kocap.cuda(), kosrccap_mask.cuda(), kotgt_mask.cuda()
        video, video_mask = video.cuda(), video_mask.cuda()

        ko_ntokens = (kocap[:, 1:] != padding_idx).data.sum()
        en_ntokens = (encap[:, 1:] != padding_idx).data.sum()
        ntokens = ko_ntokens + en_ntokens

        out_en2ko, out_ko2en, en2ko_act_pred, ko2en_act_pred = model(encap, ensrccap_mask, entgt_mask, kocap, kosrccap_mask, kotgt_mask, video, video_mask)

        loss = loss_compute(out_en2ko, kocap[:, 1:], ko_ntokens, out_ko2en, encap[:, 1:], en_ntokens)

        # task2: act_classifier
        en2ko_act_label_ntokens = (en2ko_act_pred != padding_idx).data.sum()
        ko2en_act_label_ntokens = (ko2en_act_pred != padding_idx).data.sum()

        en2ko_act_pred = en2ko_act_pred.view(en2ko_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
        ko2en_act_pred = ko2en_act_pred.view(ko2en_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
        act_labels = act_labels.view(-1).cuda()
        en2ko_label_loss = act_label_loss_compute(en2ko_act_pred, act_labels) / en2ko_act_label_ntokens.float()
        ko2en_label_loss = act_label_loss_compute(ko2en_act_pred, act_labels) / ko2en_act_label_ntokens.float()
        label_loss = en2ko_label_loss + ko2en_label_loss

        loss = loss + args.label_l * label_loss

        if epoch + 1 > args.forward_steps:
            # generate tgt - greedy search
            hypotheses_en2ko = generate_tgt(out_en2ko, model.ko_generator, max_len=args.maxlen)
            hypotheses_en2ko, hypotheses_en2ko_mask = tok_ko.encode_encodings(hypotheses_en2ko)   # pseudo_ko

            hypotheses_ko2en = generate_tgt(out_ko2en, model.en_generator, max_len=args.maxlen)
            hypotheses_ko2en, hypotheses_ko2en_mask = tok_en.encode_encodings(hypotheses_ko2en)   # pseudo_en

            hypotheses_en2ko, hypotheses_en2ko_mask, hypotheses_ko2en, hypotheses_ko2en_mask = hypotheses_en2ko.long().cuda(), hypotheses_en2ko_mask.cuda(), hypotheses_ko2en.long().cuda(), hypotheses_ko2en_mask.cuda()
            back_out_en2ko, back_out_ko2en, back_en2ko_act_pred, back_ko2en_act_pred = model(hypotheses_ko2en, hypotheses_ko2en_mask, entgt_mask, hypotheses_en2ko, hypotheses_en2ko_mask, kotgt_mask, video, video_mask)
            backward_loss = loss_compute(back_out_ko2en, kocap[:, 1:], ko_ntokens, back_out_en2ko, encap[:, 1:], en_ntokens)

            # task2: act_classifier
            back_en2ko_act_label_ntokens = (back_en2ko_act_pred != padding_idx).data.sum()
            back_ko2en_act_label_ntokens = (back_ko2en_act_pred != padding_idx).data.sum()

            back_en2ko_act_pred = back_en2ko_act_pred.view(back_en2ko_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            back_ko2en_act_pred = back_ko2en_act_pred.view(back_ko2en_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            act_labels = act_labels.view(-1).cuda()
            back_en2ko_label_loss = act_label_loss_compute(back_en2ko_act_pred, act_labels) / back_en2ko_act_label_ntokens.float()
            back_ko2en_label_loss = act_label_loss_compute(back_ko2en_act_pred, act_labels) / back_ko2en_act_label_ntokens.float()
            backward_label_loss = back_en2ko_label_loss + back_ko2en_label_loss

            backward_loss = backward_loss + args.label_l * backward_label_loss

            loss = loss + backward_loss

        loss.backward()
        loss_compute.opt.step()
        loss_compute.opt.optimizer.zero_grad()

        # Keep track of metrics
        total_loss += loss.item() * ntokens.float()
        total_tokens += ntokens

    return total_loss / total_tokens.float()


def validate(val_loader, model, loss_compute, epoch, act_label_loss_compute):
    '''
    Performs one epoch's validation.
    '''
    model.eval()  # eval mode (no dropout or batchnorm)

    ko_references = list()
    ko_hypotheses_cal = list()
    en_references = list()
    en_hypotheses_cal = list()

    total_tokens = 0
    total_loss = 0
    tokens = 0

    with torch.no_grad():
        # Batches
        for (encap, kocap), (ensrccap_mask, kosrccap_mask), (kotgt_mask, entgt_mask), video, video_mask, _, enrefs, korefs, act_labels in tqdm(val_loader, desc="epoch {}".format(epoch + 1)):

            encap, ensrccap_mask, entgt_mask, kocap, kosrccap_mask, kotgt_mask = encap.cuda(), ensrccap_mask.cuda(), entgt_mask.cuda(), kocap.cuda(), kosrccap_mask.cuda(), kotgt_mask.cuda()
            video, video_mask = video.cuda(), video_mask.cuda()

            # Forward prop.
            out_en2ko, out_ko2en, en2ko_act_pred, ko2en_act_pred = model(encap, ensrccap_mask, entgt_mask, kocap, kosrccap_mask, kotgt_mask, video, video_mask)

            ko_ntokens = (kocap[:, 1:] != padding_idx).data.sum()
            en_ntokens = (encap[:, 1:] != padding_idx).data.sum()
            ntokens = ko_ntokens + en_ntokens
            loss = loss_compute(out_en2ko, kocap[:, 1:], ko_ntokens, out_ko2en, encap[:, 1:], en_ntokens)

            # task2: act_classifier
            en2ko_act_label_ntokens = (en2ko_act_pred != padding_idx).data.sum()
            ko2en_act_label_ntokens = (ko2en_act_pred != padding_idx).data.sum()

            en2ko_act_pred = en2ko_act_pred.view(en2ko_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            ko2en_act_pred = ko2en_act_pred.view(ko2en_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            act_labels = act_labels.view(-1).cuda()
            en2ko_label_loss = act_label_loss_compute(en2ko_act_pred, act_labels) / en2ko_act_label_ntokens.float()
            ko2en_label_loss = act_label_loss_compute(ko2en_act_pred, act_labels) / ko2en_act_label_ntokens.float()
            label_loss = en2ko_label_loss + ko2en_label_loss

            loss = loss + label_loss

            total_loss += loss.item()
            total_tokens += ntokens
            tokens += ntokens

            # Hypotheses
            # en2ko
            ko_hypotheses = generate_tgt(out_en2ko, model.ko_generator, max_len=args.maxlen)
            for h in ko_hypotheses:
                ko_hypotheses_cal.append(h)
            korefs = [list(map(int, i.split())) for i in korefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]
            for r in korefs:
                ko_references.append([r])
            assert len(ko_references) == len(ko_hypotheses_cal)

            # ko2en
            en_hypotheses = generate_tgt(out_ko2en, model.en_generator, max_len=args.maxlen)
            for h in en_hypotheses:
                en_hypotheses_cal.append(h)
            enrefs = [list(map(int, i.split())) for i in enrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]
            for r in enrefs:
                en_references.append([r])
            assert len(en_references) == len(en_hypotheses_cal)

        # Calculate metrics
        avg_loss = total_loss / total_tokens.float()

        # en2ko
        corpbleu_en2ko = corpus_bleu(ko_references, ko_hypotheses_cal)
        # ko2en
        corpbleu_ko2en = corpus_bleu(en_references, en_hypotheses_cal)

    return avg_loss, corpbleu_en2ko, corpbleu_ko2en


def eval(test_loader, model, cp_file, tok_ko, tok_en, nbest=1, result_path=None):
    '''
    Testing the model
    '''
    ### the best model is the last model saved in our implementation
    epoch = torch.load(cp_file)['epoch']
    logging.info('Use epoch {0} as the best model for testing'.format(epoch))
    model.load_state_dict(torch.load(cp_file)['state_dict'])

    model.eval()  # eval mode (no dropout or batchnorm)

    en2ko_hypotheses = list()  # hypotheses (predictions)
    en2ko_references = list()
    ko2en_hypotheses = list()  # hypotheses (predictions)
    ko2en_references = list()

    # generate sentences
    start_time = time.time()

    en2ko_hypotheses_eval = list()
    ko2en_hypotheses_eval = list()
    ids = list()

    with torch.no_grad():
        # Batches
        for cnt, ((encap, kocap), (ensrccap_mask, kosrccap_mask), video, video_mask, sent_id, _, srcrefs, tgtrefs) in enumerate(test_loader):
        # for cnt, ((encap, kocap), (ensrccap_mask, kosrccap_mask), video, video_mask, sent_id, _, srcrefs, tgtrefs) in enumerate(test_loader, 1):
            encap, ensrccap_mask = encap.cuda(), ensrccap_mask.cuda()
            kocap, kosrccap_mask = kocap.cuda(), kosrccap_mask.cuda()
            video, video_mask = video.cuda(), video_mask.cuda()

            vid = sent_id[0]
            ids.append(vid)

            # Forward prop.

            # en2ko
            pred_out, _ = beam_search_decode(model, encap, ensrccap_mask, video, video_mask, args.maxlen, start_symbol=sos_idx,
                                             unk_symbol=unk_idx, end_symbol=eos_idx,
                                             pad_symbol=padding_idx)

            for n in range(min(nbest, len(pred_out))):
                pred = pred_out[n]
                temp_preds = []
                for w in pred[0]:
                    if w == eos_idx:
                        break
                    temp_preds.append(w)
                if n == 0:
                    en2ko_hypotheses_eval.append(temp_preds)
                    preds = tok_ko.decode_sentence(temp_preds)
                    en2ko_hypotheses.append(preds)

                    tgtrefs = [list(map(int, i.split())) for i in tgtrefs]

                    for r in tgtrefs:
                        en2ko_references.append([r])

                    assert len(en2ko_references) == len(en2ko_hypotheses_eval)

            # ko2en
            pred_out, _ = beam_search_decode(model, kocap, kosrccap_mask, video, video_mask, args.maxlen, start_symbol=sos_idx,
                                             unk_symbol=unk_idx, end_symbol=eos_idx,
                                             pad_symbol=padding_idx, type='ko2en')

            for n in range(min(nbest, len(pred_out))):
                pred = pred_out[n]
                temp_preds = []
                for w in pred[0]:
                    if w == eos_idx:
                        break
                    temp_preds.append(w)
                if n == 0:
                    ko2en_hypotheses_eval.append(temp_preds)
                    preds = tok_en.decode_sentence(temp_preds)
                    ko2en_hypotheses.append(preds)

                    srcrefs = [list(map(int, i.split())) for i in srcrefs]

                    for r in srcrefs:
                        ko2en_references.append([r])

                    assert len(ko2en_references) == len(ko2en_hypotheses_eval)

        ## save to json for submission
        en2ko_dc = dict(zip(ids, en2ko_hypotheses))
        print(len(en2ko_dc))
        ko2en_dc = dict(zip(ids, ko2en_hypotheses))
        print(len(ko2en_dc))
        print('========time consuming : {}========'.format(time.time() - start_time))

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(result_path + 'result_en2ko.json', 'w') as fp:
            json.dump(en2ko_dc, fp, indent=4, ensure_ascii=False)
        with open(result_path + 'result_ko2en.json', 'w') as fp:
            json.dump(ko2en_dc, fp, indent=4, ensure_ascii=False)

        en2ko_corpbleu = corpus_bleu(en2ko_references, en2ko_hypotheses_eval)
        ko2en_corpbleu = corpus_bleu(ko2en_references, ko2en_hypotheses_eval)
        logging.info('test_data: en2ko_corpbleu:{} ko2en_corpbleu {}'.format(en2ko_corpbleu, ko2en_corpbleu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='circle_ITN')
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml
        args = Arguments(yaml.full_load(fin))
    main(args)