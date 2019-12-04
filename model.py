from ALBERT.model.modeling_albert import AlbertModel, AlbertConfig
from torch import nn


class QAModel(nn.Module):
    def __init__(self, encoder_model_path, encoder_config, decoder_config, is_training = True):
        super().__init__()
        self.encoder_config = AlbertConfig(**encoder_config)
        if is_training:
            self.encoder = AlbertModel.from_pretrained(encoder_model_path, config=self.encoder_config)
        else:
            self.encoder = AlbertModel(config=self.encoder_config)
        self.decoder = nn.Linear(encoder_config["hidden_size"], decoder_config["num_label"])
        self.ner_decoder = nn.Linear(encoder_config["hidden_size"], 3)

    def forward(self, token_ids, token_segs, pos_ids, mask):
        x = self.encoder(token_ids, mask, token_segs, pos_ids)[0]
        # x -> [Batch, SeqLen, Hidden]
        qa_output = self.decoder(x)
        ner_output = self.ner_decoder(x)
        # qa_output -> [Batch, SeqLen, 2]
        # ner_decoder -> [Batch, SeqLen, 3]
        qa_output = qa_output.permute(2,0,1)
        # qa_output -> [2, Batch, SeqLen]
        start = qa_output[0]
        end = qa_output[1]
        return (start, end, ner_output)
