from ALBERT.model.modeling import ALBERT
from torch import nn


class QAModel(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = ALBERT(**encoder_config)
        self.decoder = nn.Linear(encoder_config["hidden_size"], decoder_config["num_label"])

    def forward(self, token_ids, token_segs, pos_ids, mask):
        x = self.encoder(token_ids, token_segs, pos_ids, mask)[0]
        # x -> [Batch, SeqLen, Hidden]
        output = self.decoder(x)
        # output -> [Batch, SeqLen, 2]
        output = output.permute(2,0,1)
        start = output[0]
        end = output[1]
        return (start, end)


