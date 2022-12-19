import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

import argparse
import time
import datetime
import logging
import numpy as np
import json

import torch
import torch.nn as nn

from model import make_model
from utils import set_logger,read_vocab,write_vocab,build_vocab,Tokenizer,clip_gradient,adjust_learning_rate
from dataloader import create_split_loaders
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
cc = SmoothingFunction()
from label_smoothing import *
from utils import padding_idx, sos_idx, eos_idx, unk_idx, NoamOpt, SimpleLossCompute, beam_search_decode, generate_tgt
from tqdm import tqdm


en_input, zh_input = 'en', 'zh'


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
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH
    if clear: ## delete previous vocab
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH]:
            if os.path.exists(file):
                os.remove(file)
    # Build English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'),  TRAIN_VOCAB_EN)
    #build Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

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

    maps = {'en': args.TRAIN_VOCAB_EN, 'zh': args.TRAIN_VOCAB_ZH}
    vocab_en = read_vocab(maps[en_input])
    tok_en = Tokenizer(language=en_input, vocab=vocab_en, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_zh = read_vocab(maps[zh_input])
    tok_zh = Tokenizer(language=zh_input, vocab=vocab_zh, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size en/zh:{}/{}'.format(len(vocab_en), len(vocab_zh)))

    ## Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_en, tok_zh), args.batch_size,
                                                                 args.MAX_VID_LENGTH, (en_input, zh_input),
                                                                 num_workers=4, pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format(len(train_loader), len(val_loader), len(test_loader)))

    ## init model
    model = make_model(len(vocab_en), len(vocab_zh), N=args.nb_blocks, d_model=args.d_model, d_ff=args.d_model * 4, h=args.att_h, dropout=args.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    ## define loss
    criterion_en2zh = LabelSmoothing(size=len(vocab_zh), padding_idx=padding_idx, smoothing=args.smoothing)
    criterion_zh2en = LabelSmoothing(size=len(vocab_en), padding_idx=padding_idx, smoothing=args.smoothing)
    criterion = (criterion_en2zh, criterion_zh2en)

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
            train_loss = train(train_loader, model, SimpleLossCompute(model.en_generator, model.zh_generator, criterion, opt=model_opt), epoch, tok_zh, tok_en, label_criterion)

            val_loss, corpbleu_en2zh, corpbleu_zh2en = validate(val_loader, model, SimpleLossCompute(model.en_generator, model.zh_generator, criterion, opt=None), epoch, label_criterion)
            end_time = time.time()

            epoch_time = end_time - start_time
            total_time = end_time - zero_time

            logging.info('Total time used: %s Epoch %d time used: %s train loss: %.4f val loss: %.4f corpbleu_en2zh: %.4f corpbleu_zh2en: %.4f' % (
                str(datetime.timedelta(seconds=int(total_time))),
                epoch, str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, corpbleu_en2zh, corpbleu_zh2en))

            corpbleu = corpbleu_en2zh + corpbleu_zh2en

            if corpbleu > best_val_bleu:
                best_val_bleu = corpbleu
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(),
                                 'optimizer': model_opt.optimizer.state_dict()}, cp_file)
                best_epoch = epoch

            logging.info("Finished {0} epochs of training".format(epoch + 1))

            total_train_loss.append(train_loss)
            total_val_loss.append(val_loss)

            if earlystop_flag:
                if epoch - best_epoch >= 12:
                    break

        logging.info('Best corpus bleu score {:.4f} at epoch {}'.format(best_val_bleu, best_epoch))

        ### the best model is the last model saved in our implementation
        logging.info('************ Start eval... ************')
        eval(test_loader, model, cp_file, tok_zh, tok_en, nbest=1, result_path=result_path)


def train(train_loader, model, loss_compute, epoch, tok_zh, tok_en, act_label_loss_compute):
    '''
    Performs one epoch's training.
    '''
    model.train()
    total_tokens = 0
    total_loss = 0
    for (encap, zhcap), (ensrccap_mask, zhsrccap_mask), (zhtgt_mask, entgt_mask), video, video_mask, _, _, _, act_labels in tqdm(train_loader, desc="epoch {}/{}".format(epoch + 1, args.epochs)):
         
        encap, ensrccap_mask, entgt_mask, zhcap, zhsrccap_mask, zhtgt_mask = encap.cuda(), ensrccap_mask.cuda(), entgt_mask.cuda(), zhcap.cuda(), zhsrccap_mask.cuda(), zhtgt_mask.cuda()
        video, video_mask = video.cuda(), video_mask.cuda()

        zh_ntokens = (zhcap[:, 1:] != padding_idx).data.sum()
        en_ntokens = (encap[:, 1:] != padding_idx).data.sum()
        ntokens = zh_ntokens + en_ntokens

        out_en2zh, out_zh2en, en2zh_act_pred, zh2en_act_pred = model(encap, ensrccap_mask, entgt_mask, zhcap, zhsrccap_mask, zhtgt_mask, video, video_mask)

        loss = loss_compute(out_en2zh, zhcap[:, 1:], zh_ntokens, out_zh2en, encap[:, 1:], en_ntokens)

        # task2: act_classifier
        en2zh_act_label_ntokens = (en2zh_act_pred != padding_idx).data.sum()
        zh2en_act_label_ntokens = (zh2en_act_pred != padding_idx).data.sum()

        en2zh_act_pred = en2zh_act_pred.view(en2zh_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
        zh2en_act_pred = zh2en_act_pred.view(zh2en_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
        act_labels = act_labels.view(-1).cuda()
        en2zh_label_loss = act_label_loss_compute(en2zh_act_pred, act_labels) / en2zh_act_label_ntokens.float()
        zh2en_label_loss = act_label_loss_compute(zh2en_act_pred, act_labels) / zh2en_act_label_ntokens.float()
        label_loss = en2zh_label_loss + zh2en_label_loss

        loss = loss + args.label_l * label_loss

        if epoch + 1 > args.forward_steps:
            # generate tgt - greedy search
            hypotheses_en2zh = generate_tgt(out_en2zh, model.zh_generator, max_len=args.maxlen)
            hypotheses_en2zh, hypotheses_en2zh_mask = tok_zh.encode_encodings(hypotheses_en2zh)   # pseudo_zh

            hypotheses_zh2en = generate_tgt(out_zh2en, model.en_generator, max_len=args.maxlen)
            hypotheses_zh2en, hypotheses_zh2en_mask = tok_en.encode_encodings(hypotheses_zh2en)   # pseudo_en

            hypotheses_en2zh, hypotheses_en2zh_mask, hypotheses_zh2en, hypotheses_zh2en_mask = hypotheses_en2zh.long().cuda(), hypotheses_en2zh_mask.cuda(), hypotheses_zh2en.long().cuda(), hypotheses_zh2en_mask.cuda()
            back_out_en2zh, back_out_zh2en, back_en2zh_act_pred, back_zh2en_act_pred = model(hypotheses_zh2en, hypotheses_zh2en_mask, entgt_mask, hypotheses_en2zh, hypotheses_en2zh_mask, zhtgt_mask, video, video_mask)
            backward_loss = loss_compute(back_out_zh2en, zhcap[:, 1:], zh_ntokens, back_out_en2zh, encap[:, 1:], en_ntokens)

            # task2: act_classifier
            back_en2zh_act_label_ntokens = (back_en2zh_act_pred != padding_idx).data.sum()
            back_zh2en_act_label_ntokens = (back_zh2en_act_pred != padding_idx).data.sum()

            back_en2zh_act_pred = back_en2zh_act_pred.view(back_en2zh_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            back_zh2en_act_pred = back_zh2en_act_pred.view(back_zh2en_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            act_labels = act_labels.view(-1).cuda()
            back_en2zh_label_loss = act_label_loss_compute(back_en2zh_act_pred, act_labels) / back_en2zh_act_label_ntokens.float()
            back_zh2en_label_loss = act_label_loss_compute(back_zh2en_act_pred, act_labels) / back_zh2en_act_label_ntokens.float()
            backward_label_loss = back_en2zh_label_loss + back_zh2en_label_loss

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

    zh_references = list()
    zh_hypotheses_cal = list()
    en_references = list()
    en_hypotheses_cal = list()

    total_tokens = 0
    total_loss = 0
    tokens = 0

    with torch.no_grad():
        # Batches
        for (encap, zhcap), (ensrccap_mask, zhsrccap_mask), (zhtgt_mask, entgt_mask), video, video_mask, _, enrefs, zhrefs, act_labels in tqdm(val_loader, desc="epoch {}".format(epoch + 1)):

            encap, ensrccap_mask, entgt_mask, zhcap, zhsrccap_mask, zhtgt_mask = encap.cuda(), ensrccap_mask.cuda(), entgt_mask.cuda(), zhcap.cuda(), zhsrccap_mask.cuda(), zhtgt_mask.cuda()
            video, video_mask = video.cuda(), video_mask.cuda()

            # Forward prop.
            out_en2zh, out_zh2en, en2zh_act_pred, zh2en_act_pred = model(encap, ensrccap_mask, entgt_mask, zhcap, zhsrccap_mask, zhtgt_mask, video, video_mask)

            zh_ntokens = (zhcap[:, 1:] != padding_idx).data.sum()
            en_ntokens = (encap[:, 1:] != padding_idx).data.sum()
            ntokens = zh_ntokens + en_ntokens
            loss = loss_compute(out_en2zh, zhcap[:, 1:], zh_ntokens, out_zh2en, encap[:, 1:], en_ntokens)

            # task2: act_classifier
            en2zh_act_label_ntokens = (en2zh_act_pred != padding_idx).data.sum()
            zh2en_act_label_ntokens = (zh2en_act_pred != padding_idx).data.sum()

            en2zh_act_pred = en2zh_act_pred.view(en2zh_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            zh2en_act_pred = zh2en_act_pred.view(zh2en_act_pred.size(0) * args.MAX_VID_LENGTH, -1)
            act_labels = act_labels.view(-1).cuda()
            en2zh_label_loss = act_label_loss_compute(en2zh_act_pred, act_labels) / en2zh_act_label_ntokens.float()
            zh2en_label_loss = act_label_loss_compute(zh2en_act_pred, act_labels) / zh2en_act_label_ntokens.float()
            label_loss = en2zh_label_loss + zh2en_label_loss

            loss = loss + label_loss

            total_loss += loss.item()
            total_tokens += ntokens
            tokens += ntokens

            # Hypotheses
            # en2zh
            zh_hypotheses = generate_tgt(out_en2zh, model.zh_generator, max_len=args.maxlen)
            for h in zh_hypotheses:
                zh_hypotheses_cal.append(h)
            zhrefs = [list(map(int, i.split())) for i in zhrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]
            for r in zhrefs:
                zh_references.append([r])
            assert len(zh_references) == len(zh_hypotheses_cal)

            # zh2en
            en_hypotheses = generate_tgt(out_zh2en, model.en_generator, max_len=args.maxlen)
            for h in en_hypotheses:
                en_hypotheses_cal.append(h)
            enrefs = [list(map(int, i.split())) for i in enrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]
            for r in enrefs:
                en_references.append([r])
            assert len(en_references) == len(en_hypotheses_cal)

        # Calculate metrics
        avg_loss = total_loss / total_tokens.float()

        # en2zh
        corpbleu_en2zh = corpus_bleu(zh_references, zh_hypotheses_cal)
        # zh2en
        corpbleu_zh2en = corpus_bleu(en_references, en_hypotheses_cal)

    return avg_loss, corpbleu_en2zh, corpbleu_zh2en


def eval(test_loader, model, cp_file, tok_zh, tok_en, nbest=1, result_path=None):
    '''
    Testing the model
    '''
    ### the best model is the last model saved in our implementation
    epoch = torch.load(cp_file)['epoch']
    logging.info('Use epoch {0} as the best model for testing'.format(epoch))
    model.load_state_dict(torch.load(cp_file)['state_dict'])

    model.eval()  # eval mode (no dropout or batchnorm)

    en2zh_hypotheses = list()  # hypotheses (predictions)
    en2zh_references = list()
    zh2en_hypotheses = list()  # hypotheses (predictions)
    zh2en_references = list()

    # generate sentences
    start_time = time.time()

    en2zh_hypotheses_eval = list()
    zh2en_hypotheses_eval = list()
    ids = list()

    with torch.no_grad():
        # Batches
        for cnt, ((encap, zhcap), (ensrccap_mask, zhsrccap_mask), video, video_mask, sent_id, _, srcrefs, tgtrefs) in enumerate(test_loader):
        # for cnt, ((encap, zhcap), (ensrccap_mask, zhsrccap_mask), video, video_mask, sent_id, _, srcrefs, tgtrefs) in enumerate(test_loader, 1):
            encap, ensrccap_mask = encap.cuda(), ensrccap_mask.cuda()
            zhcap, zhsrccap_mask = zhcap.cuda(), zhsrccap_mask.cuda()
            video, video_mask = video.cuda(), video_mask.cuda()

            vid = sent_id[0]
            ids.append(vid)

            # Forward prop.

            # en2zh
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
                    en2zh_hypotheses_eval.append(temp_preds)
                    preds = tok_zh.decode_sentence(temp_preds)
                    en2zh_hypotheses.append(preds)

                    tgtrefs = [list(map(int, i.split())) for i in tgtrefs]

                    for r in tgtrefs:
                        en2zh_references.append([r])

                    assert len(en2zh_references) == len(en2zh_hypotheses_eval)

            # zh2en
            pred_out, _ = beam_search_decode(model, zhcap, zhsrccap_mask, video, video_mask, args.maxlen, start_symbol=sos_idx,
                                             unk_symbol=unk_idx, end_symbol=eos_idx,
                                             pad_symbol=padding_idx, type='zh2en')

            for n in range(min(nbest, len(pred_out))):
                pred = pred_out[n]
                temp_preds = []
                for w in pred[0]:
                    if w == eos_idx:
                        break
                    temp_preds.append(w)
                if n == 0:
                    zh2en_hypotheses_eval.append(temp_preds)
                    preds = tok_en.decode_sentence(temp_preds)
                    zh2en_hypotheses.append(preds)

                    srcrefs = [list(map(int, i.split())) for i in srcrefs]

                    for r in srcrefs:
                        zh2en_references.append([r])

                    assert len(zh2en_references) == len(zh2en_hypotheses_eval)

        ## save to json for submission
        en2zh_dc = dict(zip(ids, en2zh_hypotheses))
        print(len(en2zh_dc))
        zh2en_dc = dict(zip(ids, zh2en_hypotheses))
        print(len(zh2en_dc))
        print('========time consuming : {}========'.format(time.time() - start_time))

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(result_path + 'result_en2zh.json', 'w') as fp:
            json.dump(en2zh_dc, fp, indent=4, ensure_ascii=False)
        with open(result_path + 'result_zh2en.json', 'w') as fp:
            json.dump(zh2en_dc, fp, indent=4, ensure_ascii=False)

        en2zh_corpbleu = corpus_bleu(en2zh_references, en2zh_hypotheses_eval)
        zh2en_corpbleu = corpus_bleu(zh2en_references, zh2en_hypotheses_eval)
        logging.info('test_data: en2zh_corpbleu:{} zh2en_corpbleu {}'.format(en2zh_corpbleu, zh2en_corpbleu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='circle_ITN')
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml
        args = Arguments(yaml.full_load(fin))
    main(args)