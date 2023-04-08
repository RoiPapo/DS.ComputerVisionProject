#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from eval import f_score, edit_score
from tqdm import tqdm


class MS_TCN2(nn.Module):

    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split,sc, kl=False,  sample_size=5):
      
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.fold = split
        self.kl = kl
        self.sc=sc
        self.kl_str = "KL" if self.kl else "NoKL"
        self.sample_size = sample_size
        self.sample_size_str = f"SampleSize{sample_size}"
        ext = [self.sample_size_str,"bottle neck 0"] #TODO:  add botleneckName
        self.exp_name = "-".join(ext)
        self.overlap = [.1, .25, .5]
        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def clear_ml_reporter(self, clogger, epoch, epoch_loss_train, batch_gen_train, epoch_loss_val, batch_gen_val,
                          correct_train, total_train, correct_val, total_val, edit_score_val, edit_score_train,
                          f1s_train, f1s_val):
        clogger.report_scalar(self.exp_name + " Losses - " + self.fold, "Train Loss", iteration=epoch + 1,
                              value=epoch_loss_train / len(batch_gen_train.list_of_examples))
        clogger.report_scalar(self.exp_name + " Losses - " + self.fold, "Validation Loss", iteration=epoch + 1,
                              value=epoch_loss_val / len(batch_gen_val.list_of_examples))
        clogger.report_scalar(self.exp_name + " Accuracies - " + self.fold, "Train Accuracy",
                              iteration=epoch + 1,
                              value=(float(correct_train) / total_train))
        clogger.report_scalar(self.exp_name + " Accuracies - " + self.fold, "Validation Accuracy",
                              iteration=epoch + 1,
                              value=(float(correct_val) / total_val))
        clogger.report_scalar("Validation Accuracies - " + self.fold, f"{self.sample_size} sample rate",
                              iteration=epoch + 1,
                              value=(float(correct_val) / total_val))
        clogger.report_scalar("Validation Edit Score - " + self.fold, f"{self.sample_size} sample rate",
                              iteration=epoch + 1,
                              value=np.mean(edit_score_val))
        clogger.report_scalar("Train Edit Score - " + self.fold, f"{self.sample_size} sample rate",
                              iteration=epoch + 1,
                              value=np.mean(edit_score_train))
        for k in range(len(self.overlap)):
            clogger.report_scalar(self.exp_name + " F1 Validation - " + self.fold,
                                  f"F1@{int(self.overlap[k] * 100)}",
                                  iteration=epoch + 1,
                                  value=(float(f1s_val[k]) / len(batch_gen_val.list_of_examples)))
            clogger.report_scalar(self.exp_name + " F1 Train - " + self.fold, f"F1@{int(self.overlap[k] * 100)}",
                                  iteration=epoch + 1,
                                  value=(float(f1s_train[k]) / len(batch_gen_train.list_of_examples)))

    def train(self, save_dir, batch_gen_train, batch_gen_val, num_epochs, batch_size, learning_rate, device, clogger):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_f1 = 0
        best_epoch = 0
        for epoch in tqdm(range(num_epochs)):
            epoch_loss_train = 0
            correct_train = 0
            total_train = 0
            f1s_train = [0, 0, 0]
            edit_score_train = []
            self.model.train()
            batch_i = 0
            batch_loss = 0
            while batch_gen_train.has_next():
                ####################### Training #######################
                batch_i += 1
                batch_input, batch_target, mask = batch_gen_train.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)
                loss = 0
                for i, p in enumerate(predictions):              
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                    batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                        min=0, max=16) * mask[:, :, 1:])
                    if self.kl:
                        loss += 0.5 * torch.mean(F.softmax(p[:, :, 1:], dim=1) * (
                                F.log_softmax(p[:, :, 1:], dim=1) - F.log_softmax(p.detach()[:, :, :-1], dim=1)))
                batch_loss += loss / 5
                epoch_loss_train += loss.item()
                if batch_i % 5 == 0:
                    batch_loss.backward()
                    optimizer.step()
                    batch_loss = 0

                _, predicted = torch.max(predictions[-1].data, 1)
                correct_train += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_train += torch.sum(mask[:, 0, :]).item()
                tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

                for s in range(len(self.overlap)):
                    tp1, fp1, fn1 = f_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist(), self.overlap[s],
                                            use_costumization=True)
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                for s in range(len(self.overlap)):
                    precision = tp[s] / float(tp[s] + fp[s])
                    recall = tp[s] / float(tp[s] + fn[s])
                    f1 = 2.0 * (precision * recall) / (precision + recall)
                    f1 = np.nan_to_num(f1) * 100
                    f1s_train[s] += f1

                edit_score_train.append(edit_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist()))
            ####################### Validation #######################
            epoch_loss_val = 0
            correct_val = 0
            total_val = 0
            f1s_val = [0, 0, 0]
            edit_score_val = []
            self.model.eval()
            while batch_gen_val.has_next():
                batch_input, batch_target, mask = batch_gen_val.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                predictions = self.model(batch_input)
                loss = 0
                for i, p in enumerate(predictions):
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                    batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                        min=0, max=16) * mask[:, :, 1:])
                    if self.kl:
                        loss += 0.5 * torch.mean(F.softmax(p[:, :, 1:], dim=1) * (
                                F.log_softmax(p[:, :, 1:], dim=1) - F.log_softmax(p.detach()[:, :, :-1], dim=1)))

                epoch_loss_val += loss.item()
                _, predicted = torch.max(predictions[-1].data, 1)
                correct_val += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_val += torch.sum(mask[:, 0, :]).item()
                tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
                for s in range(len(self.overlap)):
                    tp1, fp1, fn1 = f_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist(), self.overlap[s],
                                            use_costumization=True)
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                for s in range(len(self.overlap)):
                    precision = tp[s] / float(tp[s] + fp[s])
                    recall = tp[s] / float(tp[s] + fn[s])
                    f1 = 2.0 * (precision * recall) / (precision + recall)
                    f1 = np.nan_to_num(f1) * 100
                    f1s_val[s] += f1
                edit_score_val.append(edit_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist()))
            batch_gen_train.reset()
            batch_gen_val.reset()
            if (float(f1s_val[-1]) / len(batch_gen_val.list_of_examples)) > best_f1:
                best_f1 = (float(f1s_val[-1]) / len(batch_gen_val.list_of_examples))
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_dir + "/best.model")
                torch.save(optimizer.state_dict(), save_dir + "/best.opt")

            self.clear_ml_reporter(clogger, epoch, epoch_loss_train, batch_gen_train, epoch_loss_val, batch_gen_val,
                              correct_train, total_train, correct_val, total_val, edit_score_val, edit_score_train,
                              f1s_train, f1s_val)

            logger.info(
                "[epoch %d]: epoch loss train set = %f,   acc_train = %f" % (
                    epoch + 1, epoch_loss_train / len(batch_gen_train.list_of_examples),
                    float(correct_train) / total_train))
            logger.info(
                "[epoch %d]: epoch loss validation set = %f,   acc_val = %f" % (
                    epoch + 1, epoch_loss_val / len(batch_gen_val.list_of_examples),
                    float(correct_val) / total_val))

        logger.info(f"{self.exp_name} Run: Best Validation F1@50 = %f at epoch = %f" % (best_f1, best_epoch + 1))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/best.model"))
            list_of_vids = vid_list_file
            for vid in list_of_vids:
                # print vid
                features = np.load(features_path + vid)
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
