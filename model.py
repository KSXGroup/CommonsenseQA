import math
import torch
import collections
import numpy as np
from data.knowlege_graph_utils import *
from ALBERT.model.modeling_albert import AlbertModel, AlbertConfig
from torch import nn

# class LocalGraphAttention(nn.Module):
#     def __init__(self, encoder_config, decoder_config, device):
#         super().__init__()
#         self.device = device
#         self.hidden_size =  encoder_config["hidden_size"]
#         self.node_embedding = torch.nn.Embedding(decoder_config["node_num"], decoder_config["node_embedding_dim"])
#         self.proj = nn.Linear(decoder_config["node_embedding_dim"] * 2 ,
#                               decoder_config["node_embedding_dim"], bias=False)
#         self.scale = torch.nn.Embedding(decoder_config["relation_num"], decoder_config["node_embedding_dim"])
#         self.act = nn.LeakyReLU(negative_slope=0.2)
#         self.softmax = nn.Softmax(-1)
#
#     def forward(self, nodes_to_update, kg):
#         updated_dict = dict()
#         for node in nodes_to_update:
#             rels = [0]
#             neis = [node]
#             for rel, nei in kg[node]:
#                 rels.append(rel)
#                 neis.append(nei)
#             neighbours = self.scale(torch.tensor(neis, device=self.device))
#             relations = self.scale(torch.tensor(rels, device=self.device))
#             neigbour_embedding = self.node_embedding(neighbours)
#             self_embedding = self.node_embedding([node]).repeat(len(neis), 1)
#             projected = self.proj(torch.cat([self_embedding, neigbour_embedding], dim=1))
#             score = self.softmax(self.act(torch.sum(projected * relations, dim=1))).unsqueeze(1)
#             updated_dict[node] = torch.sum(score * neigbour_embedding, dim = 0)
#         return updated_dict
#


class BiAttention(nn.Module):
    def __init__(self, encoder_config, is_training=True):
        super().__init__()
        self.hidden_size =  encoder_config["hidden_size"]
        self.w_q = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.w_k = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.w_v = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.dense_0 = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.dense_1 = nn.Linear(encoder_config["hidden_size"], encoder_config["hidden_size"])
        self.drop = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Tanh()

    def forward(self, x, mask, divide_pos):
        res = torch.zeros_like(x)
        for i in range(res.shape[0]):
            query = x[i][:divide_pos[i]]
            context = x[i][divide_pos[i]:]
            context_mask = mask[i][divide_pos[i]:]
            qq = self.w_q(query)  # [QueryLen, Hidden]
            ck = self.w_k(context)  # [ContextLen, Hidden]
            cv = self.w_v(context)  # [ContextLen, Hidden]

            cq = self.w_q(context)  # [ContextLen, Hidden]
            qk = self.w_k(query)  # [QueryLen, Hidden]
            qv = self.w_v(query)  # [QueryLen, Hidden]

            q2c_tmp = torch.matmul(qq, ck.transpose(1, 0)) / math.sqrt(self.hidden_size)
            q2c_tmp += torch.unsqueeze(context_mask, 0).float()  # [QueryLen, ContextLen]
            q2c = torch.matmul(self.softmax(q2c_tmp), cv)  # [QueryLen, Hidden]

            c2q_tmp = torch.matmul(cq, qk.transpose(1, 0)) / math.sqrt(self.hidden_size)  # [ContextLen, QueryLen]
            c2q = torch.matmul(self.softmax(c2q_tmp), qv)  # [ContextLen, Hidden]

            res[i][:divide_pos[i]] = q2c
            res[i][divide_pos[i]:] = c2q

        return self.dense_1(self.drop(self.act(self.dense_0(res)))) + x

class QADecoder(nn.Module):
    def __init__(self, encoder_config, decoder_config, is_training = True):
        super().__init__()
        self.reasoning_hop = decoder_config["reasoning_hop"]
        self.hidden_size = encoder_config["hidden_size"]
        self.biattention_layers = nn.ModuleList()
        #self.knowledge_graph = knowledge_graph
        #self.gat_layer = LocalGraphAttention(encoder_config, decoder_config , device)
        #self.graph_proj = nn.Linear(decoder_config["node_embedding_dim"], encoder_config["hidden_size"])
        self.qa_output = nn.Linear(encoder_config["hidden_size"], decoder_config["num_label"])
        self.ner_output = nn.Linear(encoder_config["hidden_size"], decoder_config["ner_label"])

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

        for i in range(self.reasoning_hop):
            self.biattention_layers.append(BiAttention(encoder_config, is_training))

    def forward(self, x, mask, divide_pos, entity_start, entity_end, entity_node_index):

        # node_to_update = list(entity_node_index)
        # for node in entity_node_index:
        #     if node != -1:
        #         for rel, nei in self.knowledge_graph[node]:
        #             node_to_update.append(nei)

        for i in range(self.reasoning_hop):
            # graph_res = self.gat_layer(node_to_update, self.knowledge_graph)
            #
            # for i, idx in enumerate(entity_node_index):
            #     if idx != -1:
            #         x[entity_start[i]] *= self.graph_proj(graph_res[idx])

            x = self.biattention_layers[i](x, mask, divide_pos)

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

    def forward(self, token_ids, token_segs, pos_ids, mask, divide_pos, entity_start, entity_end, entity_node_index):
        x = self.encoder(token_ids, mask, token_segs, pos_ids)[0]
        # x -> [Batch, SeqLen, Hidden]
        qa_output, ner_output = self.decoder(x, mask, divide_pos, entity_start, entity_end, entity_node_index)
        # qa_output -> [Batch, SeqLen, 2]
        # ner_decoder -> [Batch, SeqLen, 3]
        qa_output = qa_output.permute(2,0,1)
        # qa_output -> [2, Batch, SeqLen]
        start = qa_output[0]
        end = qa_output[1]
        return (start, end, ner_output)
