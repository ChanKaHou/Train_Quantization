import os
import sys
import numpy
import torch
import torchvision
import torchjpeg.dct
import torchjpeg.quantization
import torchjpeg.metrics
import pytorch_lightning


class Model(pytorch_lightning.LightningModule):
    def __init__(self, Q):
        super().__init__()

        self.subsample = "422"

        self.Q = Q # [1, 100]
        if (self.Q < 50):
            self.S = 5000/self.Q
        elif (self.Q < 100):
            self.S = 200 - 2*self.Q
        else:
            self.S = 1
        self.DQT_Y = numpy.floor((self.S*16 + 50)/100)
        self.DQT_CbCr = numpy.floor((self.S*17 + 50)/100)

        self.QUAN = torch.nn.Parameter(torch.ones([1,2,8,8]))
        self.quan = torch.empty([1,2,8,8])

    def encode(self, input):
        ycbcr = torchjpeg.dct.to_ycbcr(input)
        dct = torchjpeg.dct.batch_dct(ycbcr)
        if self.subsample == "422":
            qY, qCb, qCr = torchjpeg.quantization.quantize_multichannel(dct, mat=self.quan) #4:2:2
        elif self.subsample == "444":
            qY = torchjpeg.quantization.quantize(dct[:,0:1,:,:], self.quan[:,0:1,:,:])
            qCb = torchjpeg.quantization.quantize(dct[:,1:2,:,:], self.quan[:,1:2,:,:])
            qCr = torchjpeg.quantization.quantize(dct[:,2:3,:,:], self.quan[:,1:2,:,:])
        return qY, qCb, qCr

    def ratio(self, qY, qCb, qCr):
        dY = torchjpeg.dct.zigzag(qY).view(-1,64).flip(-1).not_equal(0).cumsum(-1).not_equal(0).sum(-1)
        dCb = torchjpeg.dct.zigzag(qCb).view(-1,64).flip(-1).not_equal(0).cumsum(-1).not_equal(0).sum(-1)
        dCr = torchjpeg.dct.zigzag(qCr).view(-1,64).flip(-1).not_equal(0).cumsum(-1).not_equal(0).sum(-1)
        return torch.divide(dY.sum() + dCb.sum() + dCr.sum(), qY.numel() + qCb.numel() + qCr.numel())

    def decode(self, qY, qCb, qCr):
        if self.subsample == "422":
            dct = torchjpeg.quantization.dequantize_multichannel(qY, qCb, qCr, mat=self.quan)#4:2:2
        elif self.subsample == "444":
            dct = torch.cat([
                torchjpeg.quantization.dequantize(qY, self.quan[:,0:1,:,:]),
                torchjpeg.quantization.dequantize(qCb, self.quan[:,1:2,:,:]),
                torchjpeg.quantization.dequantize(qCr, self.quan[:,1:2,:,:])
                ], dim=1)
        ycbcr = torchjpeg.dct.batch_idct(dct)
        return torchjpeg.dct.to_rgb(ycbcr)

    def forward(self, input):
        self.quan = torch.sigmoid(self.QUAN)*127.0 + 1.0
        self.quan[0][0][0][0] = self.DQT_Y
        self.quan[0][1][0][0] = self.DQT_CbCr

        qY, qCb, qCr = self.encode(input)
        ratio = self.ratio(qY, qCb, qCr)
        rgb = self.decode(qY, qCb, qCr)

        ssim = torchjpeg.metrics.ssim(rgb, input)

        return ratio, ssim

    def prepare_data(self):
        self.trainSet = torchvision.datasets.ImageFolder(root='../../data/DIV2K/train')
        print(self.trainSet) #800

        self.valSet = torchvision.datasets.ImageFolder(root='../../data/DIV2K/val')
        print(self.valSet) #100

        self.testSet = torchvision.datasets.ImageFolder(root='../../data/rgb8bit/')
        print(self.testSet) #12

    def setup(self, stage):
        if (stage == 'fit'):
            self.trainSet.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x*255,
            ])
            self.trainData = self.trainSet

            self.valSet.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x*255,
            ])
            self.valData = self.valSet
        elif (stage == 'test'):
            self.testSet.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x*255,
            ])
            self.testData = self.testSet

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainData, batch_size=1, num_workers=os.cpu_count())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valData, batch_size=1, num_workers=os.cpu_count())

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testData, batch_size=1, num_workers=os.cpu_count())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        #return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'reduce_on_plateau': True, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        data, label = batch
        ratio, ssim = self(data)
        #loss = (self.Q/100 - ssim).pow(2.0).mean()
        loss = torch.log(torch.cosh(self.Q/100 - ssim)).mean()
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        ratio, ssim = self(data)
        #loss = (self.Q/100 - ssim).pow(2.0).mean()
        loss = torch.log(torch.cosh(self.Q/100 - ssim)).mean()
        self.log('val_ratio', ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, label = batch
        ratio, ssim = self(data)
        #loss = (self.Q/100 - ssim).pow(2.0).mean()
        loss = torch.log(torch.cosh(self.Q/100 - ssim)).mean()
        self.log('test_ratio', ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        print()
        #print(self.quan.round())

    def on_train_end(self):
        print()
        print(self.quan.round())

    def on_test_end(self):
        with open("DIV2K_Q="+str(self.Q)+".txt", "w") as file:
            print(self.quan.round(), file=file)

if __name__ == "__main__":
    model = Model(Q=int(sys.argv[1]))
    trainer = pytorch_lightning.Trainer(gpus=-1, max_epochs=100)
    trainer.fit(model)
    trainer.test(model)
