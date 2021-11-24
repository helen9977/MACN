import torch
from model import seg_model
import numpy as np
import os
import json
from transformers import BertTokenizer, BertModel, BertForMaskedLM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda')
def get_bert_input(text_path):
    with open(text_path,'r',encoding='utf8')as fp:
            json_data = json.load(fp)
            vo = ','.join(json_data['video_ocr'].split('|'))
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    token_bert = bert_tokenizer.encode_plus(vo,max_length=512,padding='max_length',truncation='only_first',)
    input_ids = torch.tensor(token_bert['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.tensor(token_bert['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.tensor(token_bert['attention_mask']).long().unsqueeze(0)
    return input_ids,token_type_ids,attention_mask
def get_va_feat(video_feat_path,audio_feat_path):
    video_feature  = torch.tensor((torch.load(video_feat_path))[:60*4])
    audio_feature = torch.tensor((np.load(audio_feat_path))[:60])

    pad_lenth1 = 60*4-len(video_feature)
    pad_lenth2 = 60-len(audio_feature)
    audio_feature = torch.cat([audio_feature.float(),torch.zeros((pad_lenth2,audio_feature.shape[1])).float()],dim=0)
    video_feature = torch.cat([video_feature,torch.zeros((pad_lenth1,video_feature.shape[1])).float()],dim=0)
    return video_feature,audio_feature
def inference(Model,video_feat_path,audio_feat_path,text_feat_path,idx_label):
    Model.eval()
    video_feature,audio_feature = get_va_feat(video_feat_path,audio_feat_path)
    input_ids,token_type_ids,attention_mask = get_bert_input(text_feat_path)
    video_feature = video_feature.unsqueeze(0).to(device)
    audio_feature = audio_feature.unsqueeze(0).to(device)
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    seg_con,cls_scene,cls_window,seg_py = Model(video_feature,audio_feature,input_ids,token_type_ids,attention_mask)
    seg_py = torch.sigmoid(seg_py).squeeze(0)
    predict_con = seg_con.squeeze(0)
    cls_window = torch.sigmoid(cls_window).view(60,82).detach().cpu().numpy()
    predict_con = torch.sigmoid(predict_con).view(60).detach().cpu().numpy()
    cls_scene = torch.sigmoid(cls_scene).view(60,60,82).detach().cpu().numpy()
    seg_pre = []
    cls_pre = []
    score_pre = []
    begin = 0
    predict = {}
    for j in range(60):
        if predict_con[j]>0.5:
            cls_pre_cur = []
            seg_pre.append([begin,j+seg_py[j].detach().item()])                        

            ac = (cls_window[int(begin):j]).mean(0)
            top_score = []
            score = []
            for n,c in enumerate(cls_scene[int(begin),j]+ac):
                cls_pre_cur.append(idx_label[n])
                score.append(c.item()/2)
            score = np.array(score)

            top_index = score.argsort()[-20:][::-1]
            score = score[top_index].tolist()
            cls_pre_cur = [cls_pre_cur[i] for i in top_index]
            cls_pre.append(cls_pre_cur)
            score_pre.append(score)

            begin = j+seg_py[j].detach().item()
    predict['annotations'] = [{'segment':ls,'labels':cls_pre[i],'scores':score_pre[i]} for i,ls in enumerate(seg_pre)]
    return predict
def main():
    
    
    
    Model = seg_model(1536,'gru',7,3,4).to(device)
    Model.load_state_dict(torch.load('MACN_Weight.pkl'))
    video_feat_path1 = './Data/4bff6d505cd887f5059490f6ae5758f5.pkl'
    audio_feat_path1 = './Data/4bff6d505cd887f5059490f6ae5758f5.npy'
    text_feat_path1 = './Data/4bff6d505cd887f5059490f6ae5758f5.txt'

    video_feat_path2 = './Data/d8863af5703bd614b6c01e2ba4decd0a.pkl'
    audio_feat_path2 = './Data/d8863af5703bd614b6c01e2ba4decd0a.npy'
    text_feat_path2 = './Data/d8863af5703bd614b6c01e2ba4decd0a.txt'
    label_idx = {}
    idx_label = {}
    with open('./Data/label_id.txt','r',encoding='UTF-8' )as f:
        for x in f.readlines():
            label_idx[x.split()[0]] = int(x.split()[1])
            
            idx_label [int(x.split()[1])] = x.split()[0]
            
            
    predict1 = inference(Model,video_feat_path1,audio_feat_path1,text_feat_path1,idx_label)
    predict2 = inference(Model,video_feat_path2,audio_feat_path2,text_feat_path2,idx_label)

    with open("./output/pre_4bff6d505cd887f5059490f6ae5758f5.json","w") as f:
        json.dump(predict1,f)
        print("done...")
    with open("./output/pre_d8863af5703bd614b6c01e2ba4decd0a.json","w") as f:
        json.dump(predict2,f)
        print("done...")
if __name__ == '__main__':
    main()
    # print(__name__)
