from __future__ import print_function, absolute_import
import time
import collections
import torch
import torch.nn.functional as F
from .utils.meters import AverageMeter


class GLTrainer(object):
    """Global-Local Contrastive Learning for re-ID"""

    def __init__(self, encoder, temperature=0.05):
        super(GLTrainer, self).__init__()
        self.encoder = encoder
        self.temperature = temperature

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_g = AverageMeter()
        losses_b = AverageMeter()
        losses_tm = AverageMeter()
        losses_tlr = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            imgs1, imgs2, labels, indexes = self._parse_data(inputs)
            batch_size = imgs1.size(0)
            imgs = torch.cat([imgs1, imgs2], dim=0)

            # forward
            f_out, f_b, f_tl, f_tm, f_tr = self._forward(imgs)  # 256*2048
            z = F.normalize(f_out, dim=1)
            z_b = F.normalize(f_b, dim=1)
            z_tl = F.normalize(f_tl, dim=1)
            z_tm = F.normalize(f_tm, dim=1)
            z_tr = F.normalize(f_tr, dim=1)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss_g = self.contrastive_loss(z, labels, batch_size)
            loss_b = self.contrastive_loss(z_b, labels, batch_size)
            loss_tm = self.contrastive_loss(z_tm, labels, batch_size)
            loss_tlr = self.cross_contrastive_loss(z_tl, z_tr, labels, batch_size)
            loss = loss_g + 0.33*loss_b + 0.33*loss_tm + 0.33*loss_tlr


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_g.update(loss_g.item())
            losses_b.update(loss_b.item())
            losses_tm.update(loss_tm.item())
            losses_tlr.update(loss_tlr.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss_g {:.3f} ({:.3f})\t'
                      'Loss_b {:.3f} ({:.3f})\t'
                      'Loss_tm {:.3f} ({:.3f})\t'
                      'Loss_tlr {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_g.val, losses_g.avg,
                              losses_b.val, losses_b.avg,
                              losses_tm.val, losses_tm.avg,
                              losses_tlr.val, losses_tlr.avg))

    def _parse_data(self, inputs):
        imgs1, imgs2, _, pids, _, indexes = inputs
        return imgs1.cuda(), imgs2.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def contrastive_loss(self, features, label, batch_size):
        N = batch_size
        # features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        # label = concat_all_gather(labels)
        label = label.view(-1, 1)
        label = label.repeat(2, 1)

        label_mask = label.eq(label.t()).float()
        # is_neg = 1 - label_mask

        # 2N x (2N - 1)
        pos_mask = torch.masked_select(label_mask.bool(), mask == 1).reshape(2 * N, -1)
        # neg_mask = torch.masked_select(is_neg.bool(), mask == 1).reshape(2 * N, -1)

        # rank, world_size = get_dist_info()
        # size = int(2 * N / world_size)

        # pos_mask = torch.split(pos_mask, [size] * world_size, dim=0)[rank]
        # neg_mask = torch.split(neg_mask, [size] * world_size, dim=0)[rank]
        # logit = torch.split(logit, [size] * world_size, dim=0)[rank]

        n = logit.size(0)
        loss = []

        for i in range(n):
            if label[i] == -1:
                continue
            pos_inds = torch.nonzero(pos_mask[i] == 1, as_tuple=False).view(-1)
            # neg_inds = torch.nonzero(neg_mask[i] == 1, as_tuple=False).view(-1)

            loss_single_img = []
            for j in range(pos_inds.size(0)):
                positive = logit[i, pos_inds[j]].reshape(1, 1)
                logit_i = logit[i]
                negative = torch.cat((logit_i[0:pos_inds[j]], logit_i[pos_inds[j]+1:]), dim=0).unsqueeze(0)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= 0.05
                _label = _logit.new_zeros((1,), dtype=torch.long)
                _loss = torch.nn.CrossEntropyLoss()(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss_ = sum(loss)
        # loss /= logit.size(0)
        loss_ /= len(loss)
        return loss_

    def cross_contrastive_loss(self, feats_tl, feats_tr, label, batch_size):
        N = batch_size
        # features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit_ll = torch.matmul(feats_tl, feats_tl.t())
        logit_rr = torch.matmul(feats_tr, feats_tr.t())
        temp1 = torch.where(logit_ll > logit_rr, logit_ll, logit_rr)
        logit_lr = torch.matmul(feats_tl, feats_tr.t())
        temp2 = torch.where(temp1 > logit_lr, temp1, logit_lr)
        logit_rl = torch.matmul(feats_tr, feats_tl.t())
        logit = torch.where(temp2 > logit_rl, temp2, logit_rl)

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        # label = concat_all_gather(labels)
        label = label.view(-1, 1)
        label = label.repeat(2, 1)

        label_mask = label.eq(label.t()).float()
        # is_neg = 1 - label_mask

        # 2N x (2N - 1)
        pos_mask = torch.masked_select(label_mask.bool(), mask == 1).reshape(2 * N, -1)
        # neg_mask = torch.masked_select(is_neg.bool(), mask == 1).reshape(2 * N, -1)

        # rank, world_size = get_dist_info()
        # size = int(2 * N / world_size)

        # pos_mask = torch.split(pos_mask, [size] * world_size, dim=0)[rank]
        # neg_mask = torch.split(neg_mask, [size] * world_size, dim=0)[rank]
        # logit = torch.split(logit, [size] * world_size, dim=0)[rank]

        n = logit.size(0)
        loss = []

        for i in range(n):
            if label[i] == -1:
                continue
            pos_inds = torch.nonzero(pos_mask[i] == 1, as_tuple=False).view(-1)
            # neg_inds = torch.nonzero(neg_mask[i] == 1, as_tuple=False).view(-1)

            loss_single_img = []
            for j in range(pos_inds.size(0)):
                positive = logit[i, pos_inds[j]].reshape(1, 1)
                logit_i = logit[i]
                negative = torch.cat((logit_i[0:pos_inds[j]], logit_i[pos_inds[j]:]), dim=0).unsqueeze(0)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= 0.05
                _label = _logit.new_zeros((1,), dtype=torch.long)
                _loss = torch.nn.CrossEntropyLoss()(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss_ = sum(loss)
        # loss /= logit.size(0)
        loss_ /= len(loss)
        return loss_