"""
    RIS-GAN
    func  : 模型训练
    Author: Chen Yu
    Date  : 2020.10.20
"""
import os

import torchvision
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F

from vgg import vgg16
from net import Generator, Discriminator
from dataset import IstdDataset


L_1 = 10
L_2 = 100
L_3 = 1
L_4 = 1
B_1 = 0.1
B_2 = 0.2


class Trainer(object):

    def __init__(self, batch_size=8, num_workers=4, device='cuda'):
        self.batch_size = batch_size
        self.device = device
        self.e = 0
        self.vgg = vgg16().to(self.device)
        self.vgg.eval()
        for p in self.vgg.parameters():  # reset requires_grad
            p.requires_grad_(False)

        self.generator_residual = Generator(in_channel=3).to(self.device)
        self.generator_illuminate = Generator(in_channel=3).to(self.device)
        self.generator_finial = Generator(in_channel=9).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.lr = 1e-3
        self.optimizer_residual = torch.optim.SGD(self.generator_residual.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer_illuminate = torch.optim.SGD(self.generator_illuminate.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer_finial = torch.optim.SGD(self.generator_finial.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr * 0.1, betas=(0.5, 0.999))

        self.train_ds = DataLoader(IstdDataset('ISTD_Dataset/train'), batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)
        self.test_ds = DataLoader(IstdDataset('ISTD_Dataset/test'), batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)

        self.one = torch.FloatTensor([1]).to(device)
        self.mone = self.one * -1

    # def reinit_d(self):
    #     """ Re-initialize the weights of netD
    #     """
    #     self.discriminator.apply(weights_init)
    #     print('   Reloading net d')

    def eval(self, e, sd):
        self.generator_finial.eval()
        self.generator_residual.eval()
        self.generator_illuminate.eval()

        fake_res = self.generator_residual(sd)
        fake_illum = self.generator_illuminate(sd)
        x = torch.cat([sd, fake_res, fake_illum], dim=1)
        fake_srd = self.generator_finial(x)
        fake_srd = (fake_srd * 0.5 + 0.5)
        save_dir = os.path.join("examples_new")
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(fake_srd, os.path.join(save_dir, "samples_{}.png".format(e)),
                                     nrow=self.batch_size, padding=2)

    def generator_loss(self, sd, srd, illum_gt, res_gt,
                       fake_srd, fake_illum, fake_res,
                       gan_fake_srd, gan_fake_illum, gan_fake_res,
                       vgg_srd, vgg_fake_srd):
        loss_vis = F.l1_loss(srd, fake_srd)
        loss_percept = F.mse_loss(vgg_fake_srd, vgg_srd)
        loss_rem = loss_vis + B_1 * loss_percept

        loss_res = F.l1_loss(res_gt, fake_res)
        loss_illum = F.l1_loss(illum_gt, fake_illum)
        loss_cross = F.l1_loss(srd, sd + fake_res) + B_2 * F.l1_loss(srd, sd * fake_illum)
        loss_adv = -(gan_fake_srd + gan_fake_illum + gan_fake_res)
        loss = L_1 * loss_res + L_2 * loss_rem + L_3 * loss_illum + L_4 * loss_cross + loss_adv
        return loss

    def update_generator(self, sd, srd, res_gt, illum_gt):
        self.generator_finial.train()
        self.generator_illuminate.train()
        self.generator_residual.train()
        self.discriminator.train()

        self.generator_residual.zero_grad()
        self.generator_illuminate.zero_grad()
        self.generator_finial.zero_grad()

        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        fake_res = self.generator_residual(sd)
        fake_illum = self.generator_illuminate(sd)
        x = torch.cat([sd, fake_res, fake_illum], dim=1)
        fake_srd = self.generator_finial(x)

        gan_fake_res = self.discriminator(fake_res).mean()
        gan_fake_illum = self.discriminator(fake_illum).mean()
        gan_fake_srd = self.discriminator(fake_srd).mean()

        vgg_srd_input = F.upsample(srd, (228, 228), mode='nearest')
        vgg_fake_srd_input = F.upsample(fake_srd, (228, 228), mode='nearest')
        vgg_srd = self.vgg(vgg_srd_input)
        vgg_fake_srd = self.vgg(vgg_fake_srd_input)

        g_loss = self.generator_loss(sd, srd, illum_gt, res_gt,
                                     fake_srd, fake_illum, fake_res,
                                     gan_fake_srd, gan_fake_illum, gan_fake_res,
                                     vgg_srd, vgg_fake_srd)
        g_loss.backward(self.one)
        self.optimizer_residual.step()
        self.optimizer_illuminate.step()
        self.optimizer_finial.step()
        return g_loss

    def discriminator_loss(self, gan_real_srd, gan_real_illum, gan_real_res,
                           gan_fake_srd, gan_fake_illum, gan_fake_res):
        loss_srd = -(gan_real_srd - gan_fake_srd)
        loss_illum = -(gan_real_illum - gan_fake_illum)
        loss_res = -(gan_real_res - gan_fake_res)
        return loss_srd + loss_illum + loss_res

    def update_discriminator(self, sd, srd, res_gt, illum_gt):
        self.generator_finial.train()
        self.generator_residual.train()
        self.generator_illuminate.train()
        self.discriminator.train()
        self.discriminator.zero_grad()

        for p in self.discriminator.parameters():  # reset requires_grad
            p.requires_grad_(True)

        fake_res = self.generator_residual(sd).detach()
        fake_illum = self.generator_illuminate(sd).detach()
        x = torch.cat([sd, fake_res, fake_illum], dim=1)
        fake_srd = self.generator_finial(x).detach()

        gan_real_srd = self.discriminator(srd).mean()
        gan_real_res = self.discriminator(res_gt).mean()
        gan_real_illum = self.discriminator(illum_gt).mean()
        gan_fake_srd = self.discriminator(fake_srd).mean()
        gan_fake_res = self.discriminator(fake_res).mean()
        gan_fake_illum = self.discriminator(fake_illum).mean()
        d_loss = self.discriminator_loss(gan_real_srd, gan_real_illum, gan_real_res,
                                         gan_fake_srd, gan_fake_illum, gan_fake_res)
        d_loss.backward(self.one)
        self.optimizer_discriminator.step()
        return d_loss

    def save(self, e):
        save_dir = os.path.join("param_new")
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'gen_res': self.generator_residual.state_dict(),
            'gen_illum': self.generator_illuminate.state_dict(),
            'gen_finial': self.generator_finial.state_dict(),
            'discrimator': self.discriminator.state_dict(),
            'optim_res': self.optimizer_residual.state_dict(),
            'optim_illum': self.optimizer_illuminate.state_dict(),
            'optim_filial': self.optimizer_finial.state_dict(),
            'optim_dis': self.optimizer_discriminator.state_dict(),
            'epoch': e
        }
        path = os.path.join(save_dir, "checkpoint_%d.pkl" % e)
        torch.save(checkpoint, path)

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.generator_residual.load_state_dict(checkpoint["gen_res"])
        self.generator_illuminate.load_state_dict(checkpoint["gen_illum"])
        self.generator_finial.load_state_dict(checkpoint["gen_finial"])
        self.discriminator.load_state_dict(checkpoint["discrimator"])
        self.e = checkpoint["epoch"]
        self.optimizer_residual.load_state_dict(checkpoint["optim_res"])
        self.optimizer_illuminate.load_state_dict(checkpoint["optim_illum"])
        self.optimizer_finial.load_state_dict(checkpoint["optim_filial"])
        self.optimizer_discriminator.load_state_dict(checkpoint["optim_dis"])

    def train(self, epoch=10000):
        for e in range(self.e, epoch):
            for step, (sd, srd, res_gt, illum_gt) in enumerate(self.train_ds):
                sd = sd.to(self.device)
                srd = srd.to(self.device)
                res_gt = res_gt.to(self.device)
                illum_gt = illum_gt.to(self.device)
                g_loss = self.update_generator(sd, srd, res_gt, illum_gt)
                d_loss = self.update_discriminator(sd, srd, res_gt, illum_gt)

                if step % 5 == 0 and step != 0:
                    print("epoch %d  step % d  g_loss %.4f d_loss %.4f" % (e, step, g_loss, d_loss))
                # if torch.abs(d_loss) > 100:
                #     self.reinit_d()
            self.save(0)
            self.eval(e, sd)
            if e % 10 == 0:
                self.save(e)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load('param_new/checkpoint_450.pkl')
    trainer.train()









