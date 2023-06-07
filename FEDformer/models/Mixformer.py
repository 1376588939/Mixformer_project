import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, ConvLayer, my_Layernorm, \
    series_decomp, series_decomp_multi


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        # self.ELU = nn.ELU()
        self.dropout = nn.Dropout(configs.dropout)

        # Decompsition Kernel Size
        kernel_size = configs.moving_avg
        self.decompsition = series_decomp_multi(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # projection
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.d_ff // 16, kernel_size=1, bias=False)
        # self.conv4 = nn.Conv1d(in_channels=configs.d_ff // 16, out_channels=configs.enc_in, kernel_size=1, bias=False)
        self.projection = nn.Linear(self.d_model, self.enc_in)
        self.projection1 = nn.Linear(self.seq_len, self.pred_len)
        

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # DLinear as Decoder
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
            # self.Linear_Seasonal_1 = nn.Linear(self.seq_len, self.d_model)
            # self.Linear_Trend_1 = nn.Linear(self.seq_len, self.d_model)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x_enc, x_mark_enc, enc_self_mask=None):

        emb_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(emb_out, attn_mask=enc_self_mask)
        enc_out = emb_out + self.dropout(enc_out)
        y = self.dropout(self.activation(self.conv1(enc_out.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # proj_out = self.ELU(enc_out + y)
        # proj_out = self.projection(y)

        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

            # Use this to replace MLP in decomposition
            # y1 = self.dropout(self.activation(self.conv3(seasonal_output)))
            # seasonal_output = self.dropout(self.conv4(y1))
            # y2 = self.dropout(self.activation(self.conv3(trend_output)))
            # trend_output = self.dropout(self.conv4(y2))

        # dec_out = proj_out + seasonal_output.permute(0, 2, 1) + trend_output.permute(0, 2, 1)
        y = self.projection(y)
        y = self.projection1(y.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = y + seasonal_output.permute(0,2,1) + trend_output.permute(0,2,1)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
