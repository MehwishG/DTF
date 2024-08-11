import torch
import torch.nn as nn
from einops import rearrange
from model.module.trans import Transformer as MVG_Encoder
from model.module.trans_views import Transformer as MVG_Views


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## MHG
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        #self.norm_3 = nn.LayerNorm(args.frames)

        self.MVG_Encoder_1 = MVG_Encoder(4, args.frames, args.frames * 2, length=3 * args.n_joints, h=9)
        self.MVG_Encoder_2 = MVG_Encoder(4, args.frames, args.frames * 2, length=3 * args.n_joints, h=9)
        #self.MVG_Encoder_3 = MVG_Encoder(4, args.frames, args.frames * 2, length=2 * args.n_joints, h=9)

        ## Embedding
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(3 * args.n_joints, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(3 * args.n_joints, args.channel, kernel_size=1)
            #self.embedding_3 = nn.Conv1d(2 * args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(3 * args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(3 * args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            
        ## SRM & IFM
        self.MVG_Views = MVG_Views(args.layers, args.channel, args.d_hid, length=args.frames)

        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel * 2, momentum=0.1),
            nn.Conv1d(args.channel * 2, 3 * args.out_joints, kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape
        # print(x.shape) #torch.Size([256, 351, 17, 2])
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()
        # print(x.shape)#torch.Size([256, 34, 351])
        ## MVG
        
        x_1 = x + self.MVG_Encoder_1(self.norm_1(x))
        x_2 = x_1 + self.MVG_Encoder_2(self.norm_2(x_1))
        ## Embedding
        
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous()
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        ## SRM & IFM

        x = self.MVG_Views(x_1, x_2)
        # print("after refinement ", x.shape)# torch.Size([256, 351, 1536])
        ## Regression
        x = x.permute(0, 2, 1).contiguous()

        x = self.regression(x)
        #print("regresssion",x.shape) #torch.Size([256, 51, 351])
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()
        # print("out", x.shape) #torch.Size([256, 351, 17, 3])
        return x






