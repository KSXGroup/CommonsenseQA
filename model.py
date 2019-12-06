import math
import torch
from ALBERT.model.modeling_albert import AlbertModel, AlbertConfig
from torch import nn

class QADecoder(nn.Module):
    def __init__(self, encoder_config, decoder_config,  is_training = True):
        super().__init__()
        self.reasoning_hop = decoder_config["reasoning_hop"]
        self.hidden_size = encoder_config["hidden_size"]
        self.w_q = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.w_k = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.w_v = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.dense_0 = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.dense_1 = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.qa_output = nn.Linear(encoder_config["hidden_size"], decoder_config["num_label"])
        self.ner_output = nn.Linear(encoder_config["hidden_size"], decoder_config["ner_label"])
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Tanh()

        def _init_weights(module):
            """ Initialize the weights.
            """
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=encoder_config["initializer_range"])
                if isinstance(module, (nn.Linear)) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_init_weights)

    def forward(self, input, mask, divide_pos):
        x = input
        for i in range(self.reasoning_hop):
            res = torch.zeros_like(x)
            for i in range(x.shape[0]):
                query = x[i][:divide_pos[i]]
                context = x[i][divide_pos[i]:]
                context_mask = mask[i][divide_pos[i]:]
                qq = self.w_q(query) #[QueryLen, Hidden]
                ck = self.w_k(context) #[ContextLen, Hidden]
                cv = self.w_v(context) #[ContextLen, Hidden]

                cq = self.w_q(context) #[ContextLen, Hidden]
                qk = self.w_k(query) #[QueryLen, Hidden]
                qv = self.w_v(query) #[QueryLen, Hidden]

                q2c_tmp = torch.matmul(qq, ck.transpose(1, 0)) / math.sqrt(self.hidden_size)
                q2c_tmp += torch.unsqueeze(context_mask, 0).float() #[QueryLen, ContextLen]
                q2c = torch.matmul(self.softmax(q2c_tmp), cv) #[QueryLen, Hidden]

                c2q_tmp = torch.matmul(cq, qk.transpose(1, 0)) / math.sqrt(self.hidden_size) #[ContextLen, QueryLen]
                c2q = torch.matmul(self.softmax(c2q_tmp), qv) #[ContextLen, Hidden]

                res[i][:divide_pos[i]] = q2c
                res[i][divide_pos[i]:] = c2q
            x = self.dense_1(self.act(self.dense_0(res))) + x

        qa_out = self.qa_output(x)
        ner_out = self.ner_output(x)
        return qa_out, ner_out



class QAModel(nn.Module):
    def __init__(self, encoder_model_path, encoder_config, decoder_config, is_training = True):
        super().__init__()
        self.encoder_config = AlbertConfig(**encoder_config)
        if is_training:
            self.encoder = AlbertModel.from_pretrained(encoder_model_path, config=self.encoder_config)
        else:
            self.encoder = AlbertModel(config=self.encoder_config)
        self.decoder = QADecoder(encoder_config, decoder_config)
        #self.ner_decoder = nn.Linear(encoder_config["hidden_size"], 3)

    def forward(self, token_ids, token_segs, pos_ids, mask, divide_pos):
        x = self.encoder(token_ids, mask, token_segs, pos_ids)[0]
        # x -> [Batch, SeqLen, Hidden]
        qa_output, ner_output = self.decoder(x, mask, divide_pos)
        # qa_output -> [Batch, SeqLen, 2]
        # ner_decoder -> [Batch, SeqLen, 3]
        qa_output = qa_output.permute(2,0,1)
        # qa_output -> [2, Batch, SeqLen]
        start = qa_output[0]
        end = qa_output[1]
        return (start, end, ner_output)
