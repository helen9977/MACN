import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda')
class transformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.trf = nn.TransformerEncoderLayer(d_model=2048,nhead=int(2048/64),dim_feedforward=4096)
    def forward(self,x):
        x = self.trf(x.transpose(0,1)).transpose(0,1)
        return x,1
class seg_model(nn.Module):
    def __init__(self,v_dim,type_feate,kernel_size,padding,stride):
        super(seg_model, self).__init__()
        self.emb1 = nn.Conv1d(v_dim,1024,kernel_size=kernel_size,padding=padding,stride=stride)
        self.emb2 = nn.Linear(128,1024)
        self.att_score = nn.Sequential(
            nn.Linear(2048, 2048 // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 8, 2048, bias=False),
            nn.Sigmoid()
        )
        if type_feate=='lstm':
            self.lstm1 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm2 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm3 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm4 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm5 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
        elif type_feate=='gru':
            self.lstm1 = nn.GRU(2048,1024,batch_first =True,bidirectional=True)
            self.lstm2 = nn.GRU(2048,1024,batch_first =True,bidirectional=True)
            self.lstm3 = nn.GRU(2048,256,batch_first =True,bidirectional=True)
            self.lstm4 = nn.GRU(2048,256,batch_first =True,bidirectional=True)
            self.lstm5 = nn.GRU(2048,1024,batch_first =True,bidirectional=True)
        elif type_feate=='transformer':
            self.lstm1 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm2 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm3 = transformer()#nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm4 = transformer()#nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
            self.lstm5 = nn.LSTM(2048,1024,batch_first =True,bidirectional=True)
        self.linear_sp = nn.Linear(2048,2048)

        self.dropout = nn.Dropout(0.1)
        self.lyn = nn.LayerNorm(2048)
        self.ln_x1 = nn.LayerNorm(v_dim)
        self.ln_x2 = nn.LayerNorm(128)
        self.lyn1 = nn.LayerNorm(2048)
        self.lyn2 = nn.LayerNorm(2048)
        self.fc_cls = nn.Linear(2048,82)
        self.fc_acc = nn.Linear(2048,82)
        self.fc_py = nn.Linear(2048,1)
        self.fc_con = nn.Linear(2048,1)
        self.fc_score = nn.Linear(2048,2048)
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert.requires_grad_(False)
        self.fc_bert = nn.Linear(768,2048)
        self.fc_bert_v = nn.Linear(768,2048)
        self.fc_va = nn.Linear(2048,2048)
        self.relu = nn.ReLU()
        self.lyn3 = nn.LayerNorm(2048)
        self.conv_ann = nn.Sequential(
                            nn.Conv2d(2048,256,1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.Conv2d(256,2048,3,padding=1),
                            nn.BatchNorm2d(2048),
                            nn.ReLU(),

                        )
        params = torch.zeros((60,512), requires_grad=True)
        self.bert_pre= nn.Parameter(params)
        self.conv_dil = dil_conv()
        self.lyn_an = nn.LayerNorm(2048)
        self.linear_bert_att1 = nn.Sequential(
                            nn.Linear(768,1024),
                            nn.ReLU(),
                            nn.Linear(1024,1),

                        )
        self.linear_bert_att2 = nn.Sequential(
                            nn.Linear(512,30),
                            nn.ReLU(),


                        )
        
        self.linear_va_att1 = nn.Sequential(
                            nn.Linear(2048,1024),
                            nn.ReLU(),
                            nn.Linear(1024,1),

                        )
        self.linear_va_att2 = nn.Sequential(
                            nn.Linear(60,30),
                            nn.ReLU(),

                        )
        self.att_fus = nn.Linear(30,240)
        self.stage1 = stage2(1)
        self.stage2 = stage2(2)
        self.stage3 = stage2(4)
        self.stage4 = stage2(8)
    def forward(self, x1,x2,input_ids,token_type_ids,attention_mask):
        x1 = self.ln_x1(x1)
        x2 = self.ln_x2(x2)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        bert_att = self.linear_bert_att1(outputs).squeeze(-1)
        bert_att = self.linear_bert_att2(bert_att)
        x1 = self.emb1(x1.transpose(1,2)).transpose(1,2)

        x2 = self.emb2(x2)

        x = torch.cat([x1,x2],dim=2)
        z_att = self.att_score(x.mean(1)).unsqueeze(1)
        x = x * z_att.expand_as(x)
#         x = x*pos
        x = self.dropout(x)
        x = self.lyn(x)
        
        va_att = self.linear_va_att1(x).squeeze(-1)
        va_att = self.linear_va_att2(va_att)
        
        
        
        muti_att = self.att_fus(bert_att+va_att)
        muti_att = torch.softmax(muti_att.view(-1,4,60),dim=1)
        
        x_bert = self.fc_bert(outputs)
        x_bert_v = self.fc_bert_v(outputs)
        x_va = self.fc_va(x)
        N,L,C = x_bert.size()
        x_bert = x_bert.view(N,L,-1,128).permute(0,2,1,3).contiguous().view(-1,L,128)
        x_bert_v = x_bert_v.view(N,L,-1,128).permute(0,2,1,3).contiguous().view(-1,L,128)
        x_va = x_va.view(N,60,-1,128).permute(0,2,1,3).contiguous().view(-1,60,128)

        att_score = torch.bmm(x_va,x_bert.transpose(1,2))
        att_score = torch.softmax(att_score,dim=-1)+torch.softmax(self.bert_pre.unsqueeze(0),dim=-1)

        att_score = att_score
        x_bert_v = torch.bmm(att_score,x_bert_v)
        x_bert_v = x_bert_v.view(N,-1,60,128).permute(0,2,1,3).contiguous().view(N,60,-1)
        x = x+x_bert_v
        N,L,C = x.size()
        
        
        x_sp = torch.cat([x[:,1:,:],torch.zeros(N,1,x.shape[2]).to(device)],dim=1)
        y = x-x_sp
        y = self.linear_sp(y)
        y,_ = self.lstm2(y)
        x =  self.conv_dil(x,muti_att)
        x,_ = self.lstm1(x)
        z = self.lyn1(x)+self.lyn2(y)

        N,L,C = z.size()

        z,_ = self.lstm5(z)
        x = self.lyn3(x)
        score = self.get_triloss(self.fc_score(x),self.fc_score(x))
        
        q,_ = self.lstm3(x)
        k,_ = self.lstm4(x)
        acc = self.fc_acc(x)
        N,L,C = q.size()
        an = self.fc_con(z).squeeze(-1)
        q = q.unsqueeze(1).expand(-1,L,-1,-1)
        k = k.unsqueeze(2).expand(-1,-1,L,-1)
        ann =q*k
        ann = ann*torch.triu(torch.ones(60, 60)).to(device).unsqueeze(-1)
        N,L,L,C = ann.size()
        ann = self.stage1(ann)+self.stage2(ann)+self.stage3(ann)+self.stage4(ann)
        return an,self.fc_cls(ann),acc,self.fc_py(z).squeeze(-1)
    def get_triloss(self,x1,x2):
        x1 = torch.nn.functional.normalize(x1,dim=-1)
        x2 = torch.nn.functional.normalize(x2,dim=-1)
        score = torch.bmm(x1,x2.transpose(1,2))
        return score
class stage2(nn.Module):
    def __init__(self,dil):
        super().__init__()
        self.indexs = []
        for i in range(dil,60+dil):
            for j in range(dil,60+dil):
                index = []
                index.append([i,j-dil])
                index.append([i,j])
                index.append([i+dil,j-dil])
                index.append([i+dil,j])
                self.indexs.append(index)
        self.indexs = torch.tensor(self.indexs)
        self.bn = nn.BatchNorm2d(2048)
        self.pad = nn.ZeroPad2d((dil,dil,dil,dil))
        self.conv =nn.Conv2d(2048,2048,1,bias=False)
        self.bn2 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()

    def forward(self,x):
        N,L,L,C = x.size()
        x = x.permute(0,3,1,2).contiguous()
        res = x
        x_conv = self.pad(x)[:,:,self.indexs[:,:,0],self.indexs[:,:,1]]
        x_conv = x_conv.permute(0,1,3,2).contiguous().view(N,-1,L,L)
        x_conv = self.conv(x_conv)
        x_conv = self.relu(self.bn2(x_conv))
        x = x_conv
        return x.permute(0,2,3,1).contiguous()
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)
#         plt.figure(figsize=(8, 6))
#         plt.imshow(attn[0][0].detach().cpu(), interpolation='nearest', cmap=plt.cm.hot,
#         )
#         plt.show()
        loss = attn-attn_mask.unsqueeze(0).unsqueeze(0)
        loss = loss*loss
        loss = loss.sum()
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, loss
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model
        self.lyn = nn.LayerNorm(d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        d_model  = self.d_model
        residual = inputs
        output = self.fc(inputs)
        return self.lyn(output + residual) 
class transformer_mutis(nn.Module):
    def __init__(self,kernel_size,d_model,d_k,n_heads):
        super().__init__()
        att_sig = []
        x = np.array(list(range(60)))

        for u in range(60):
            sig = math.sqrt(kernel_size)  # 标准差δ
            y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
            att_sig.append(torch.tensor(y_sig).unsqueeze(0))
        
        self.att_sig = torch.cat(att_sig,dim=0).to(device).float()
        self.att_sig = self.att_sig*(1/self.att_sig.sum(1))
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.pos = PoswiseFeedForwardNet(d_model,2*d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_model = d_model
        self.lyn = nn.LayerNorm(d_model)
    def forward(self,x):
        residual = x
        n_heads = self.n_heads
        d_k = self.d_k
        d_model = self.d_model
        batch_size,L,C = x.size()
        Q = self.W_Q(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = self.att_sig

        context, loss = ScaledDotProductAttention()(Q, K, V, attn_mask,d_k)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_k) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        output = self.lyn(output + residual)
        output = self.pos(output)
        return output,loss
class dil_conv(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv1d(2048*2,2048,1)#nn.Conv1d(2048,2048,kernel_size=3,dilation=1,padding=1)
        self.conv2 = nn.Conv1d(2048*2,2048,1)#nn.Conv1d(2048,2048,kernel_size=3,dilation=1,padding=1)
        self.conv3 = nn.Conv1d(2048*2,2048,1)#nn.Conv1d(2048,2048,kernel_size=3,dilation=1,padding=1)
        self.conv4 = nn.Conv1d(2048*2,2048,1)#nn.Conv1d(2048,2048,kernel_size=3,dilation=1,padding=1)
        self.q = torch.tensor([1,-1]).to(device).float()
        self.conv_out = nn.Conv1d(2048,2048,kernel_size=1)
        self.index1 = self.get_index(1)
        self.index2 = self.get_index(2)
        self.index3 = self.get_index(4)
        self.index4 = self.get_index(8)
        self.bn1 = nn.LayerNorm(2048)
        self.bn2 = nn.LayerNorm(2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU()
        self.pad1 = nn.ZeroPad2d((1,1))
        self.pad2 = nn.ZeroPad2d((2,2))
        self.pad3 = nn.ZeroPad2d((4,4))
        self.pad4 = nn.ZeroPad2d((8,8))
    def forward(self,x,muti_att):
        N,L,C = x.size()
        x = self.bn1(x)
        res = x
        x = x.transpose(1,2)
        
        x_1 = self.pad1(x)[:,:,self.index1]*self.q.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x_2 = self.pad2(x)[:,:,self.index2]*self.q.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x_3 = self.pad3(x)[:,:,self.index3]*self.q.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x_4 = self.pad4(x)[:,:,self.index4]*self.q.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x_1 = self.conv1(x_1.transpose(2,3).contiguous().view(N,-1,L).contiguous()).unsqueeze(1)
        x_2 = self.conv2(x_2.transpose(2,3).contiguous().view(N,-1,L).contiguous()).unsqueeze(1)
        x_3 = self.conv3(x_3.transpose(2,3).contiguous().view(N,-1,L).contiguous()).unsqueeze(1)
        x_4 = self.conv4(x_4.transpose(2,3).contiguous().view(N,-1,L).contiguous()).unsqueeze(1)
        x = torch.cat([x_1,x_2,x_3,x_4],dim=1)
        x = x.transpose(2,3)
        x = x*muti_att.unsqueeze(-1)
        x = x.sum(1)
        x = self.relu(x)
        x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv_out(x)
#         x = self.bn3(x)
#         x = self.relu(x)
        return x+res
    def get_index(self,dil):
        index = []
        for i in range(60):
            index.append([i,i+dil*2])
        return index

def main():
    
    Model = seg_model(1536,'gru',7,3,4).to(device)
    Model.load_state_dict(torch.load('MACN_Weight.pkl'))


if __name__ == '__main__':
    main()
    # print(__name__)