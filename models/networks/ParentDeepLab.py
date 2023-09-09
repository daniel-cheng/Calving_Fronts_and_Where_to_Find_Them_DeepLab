import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import sys
import torch.utils.model_zoo as model_zoo


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()
        self.inplanes = inplanes
        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(self.inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True)

        # Middle flow
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                            grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True,
                             grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_rates[1])
        self.bn5 = nn.BatchNorm2d(2048)

        # Init weights
        self.__init_weight()

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = torch.load('./models/networks/xception-b5690688.pth')
        # pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        #Pick subset of 3 channel input kernel if inplanes < 3
        pretrain_dict['conv1.weight'] = pretrain_dict['conv1.weight'][:, 0:self.inplanes, :, :]
        for k, v in pretrain_dict.items():
            # print(k)
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(pl.LightningModule):
    def __init__(self, hparams, metric, n_classes=21, nInputChannels=1, os=16, pretrained=True):

        super(DeepLabv3_plus, self).__init__()
        self.save_hyperparameters(hparams)

        self.n_channels_of_input = nInputChannels  # Greyscale
        self.metric = metric
        self.n_classes = n_classes
        self.output_stride = os
        self.pretrained = pretrained
        self.layers = self.make_layer_structure()

    def make_layer_structure(self):
        # Atrous Conv
        self.xception_features = Xception(self.n_channels_of_input, self.output_stride, self.pretrained)

        # ASPP
        if self.output_stride == 16:
            # rates = [1, 6, 12, 18]
            rates = [1, 2, 3, 5, 7]
        elif self.output_stride == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])
        self.aspp5 = ASPP_module(2048, 256, rate=rates[4])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.aspp_maps = (len(rates) + 1) * 256
        self.conv1 = nn.Conv2d(self.aspp_maps, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, self.n_classes, kernel_size=1, stride=1))

    def forward(self, inputs):
        x, low_level_features = self.xception_features(inputs)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp5(x)
        x6 = self.global_avg_pool(x)
        x6 = F.upsample(x6, size=x5.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(inputs.size()[-2] / 4)),
                                int(math.ceil(inputs.size()[-1] / 4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def adapt_mask(self, y):
        mask_type = torch.float32 if self.n_classes == 1 else torch.long
        y = y.type(mask_type)
        return y

    def give_prediction_for_batch(self, batch):
        x, y, x_name, y_names = batch

        # Safety check
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
                torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
            print(f"invalid input detected: x {x}, y {y}", file=sys.stderr)

        y_hat = self.forward(x)

        # Safety check
        if torch.any(torch.isnan(y_hat)) or torch.any(torch.isinf(y_hat)):
            print(f"invalid output detected: y_hat {y_hat}", file=sys.stderr)

        return y_hat

    def calc_loss(self, y_hat, y):
        return NotImplemented, NotImplemented

    def make_batch_dictionary(self, loss, metric, name_of_loss):
        return NotImplemented

    def log_metric(self, outputs, train_or_val_or_test):
        pass

    def training_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        assert x.shape[1] == self.n_channels_of_input, \
            f'Network has been defined with {self.n_channels_of_input} input channels, ' \
            f'but loaded images have {x.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
        y_hat = self.give_prediction_for_batch(batch)
        train_loss, metric = self.calc_loss(y_hat, y)

        self.log('train_loss', train_loss)
        return self.make_batch_dictionary(train_loss, metric, "loss")

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Train")

    def validation_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        y_hat = self.give_prediction_for_batch(batch)
        val_loss, metric = self.calc_loss(y_hat, y)
        self.log('val_loss', val_loss)
        return self.make_batch_dictionary(val_loss, metric, "val_loss")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Val")
        self.log('avg_loss_validation', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        y_hat = self.give_prediction_for_batch(batch)
        test_loss, metric = self.calc_loss(y_hat, y)
        self.log('test_loss', test_loss)
        return self.make_batch_dictionary(test_loss, metric, "test_loss")

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr,
                                                      cycle_momentum=False,
                                                      step_size_up=30000)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        return NotImplemented
