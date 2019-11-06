from ALBERT.model.modeling import ALBERT
from torch import nn


class QAModel(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(QAModel, self).__init__()
        self.encoder = ALBERT(**encoder_config)
        self.decoder = nn.Linear(encoder_config["hidden_size"], decoder_config["num_label"])

    def forward(self):
        pass
