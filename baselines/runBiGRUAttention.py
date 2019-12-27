import torch
import tqdm
from torch import nn, from_numpy
from torch.optim.adam import Adam
from prepro import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

PRINT_EVERY = 50

NOTATION = {'\'', '\"', ',', '.', 'n\'t'}

class Example:
    def __init__(self, tokens, start, end):
        self.tokens = tokens
        self.start = start
        self.end = end

class recordDataset(Dataset):
    def __init__(self, expamples):
        self.length = len(expamples)
        self.data = expamples
        # for i, ex in enumerate(self.data):
        #     print(ex.text_tokens[ex.start_pos: ex.end_pos+1])
        #     input()

    def __getitem__(self, item):
        idata = self.data[item].tokens
        out = np.zeros(2360, dtype=np.int32)
        mask = np.zeros(2360, dtype=np.float32)
        mask[:len(idata)] = 1.0
        out[:len(idata)] = idata
        return torch.tensor(out, dtype=torch.int64),  \
               torch.tensor(mask, dtype=torch.float), \
               torch.tensor(self.data[item].start_pos, dtype=torch.int64), \
               torch.tensor(self.data[item].end_pos, dtype=torch.int64)

    def __len__(self):
        return self.length


class BiGRUAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, input_size, hidden_size, embedding_matrix):
        super().__init__()
        self.seq_len = 2360
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding.weight.data = from_numpy(embedding_matrix.astype(np.float32))
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.output = nn.Linear(2 * hidden_size, 2 * self.seq_len)
        self.hidden = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.drop = nn.Dropout(0.1)
        self.act = nn.ReLU()
        #self.w_q = nn.Linear(hidden_size, hidden_size)
        #self.w_k = nn.Linear(hidden_size, hidden_size)
        #self.w_v = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask):
        #x batch size seqlen
        # input: (seq_len, batch, input_size)
        # output: (seq_len, batch, num_directions * hidden_size)
        # h_n: (num_directions, batch, hidden_size)
        x = self.embedding(x) #batch size, seqlen, hidden
        #print(x.shape)
        output, h_n = self.gru(x.permute(1, 0, 2))
        output = output.view(x.shape[1], x.shape[0], 2, self.hidden_size).permute(1,0,2,3)
        forward_out = output[:,:,0] #(batch, seq_len, hidden_size)
        backward_out = output[:,:,1]#(batch, seq_len, hidden_size)
        forward_hn = h_n[0].unsqueeze(2) #(batch,hidden_size,1)
        backward_hn = h_n[1].unsqueeze(2) #(batch,hidden_size,1)
        #print(forward_out.shape, forward_hn.shape)
        tmp =torch.bmm(forward_out, forward_hn).permute((0,2,1))
        #print(tmp.shape)
        #print(mask.shape)
        forward_out_score = self.softmax( tmp
                                         + mask.unsqueeze(1)) / np.sqrt(512) #(batch, 1, seq_len)
        backward_out_score = self.softmax(torch.bmm(backward_out, backward_hn).permute((0,2,1))
                                          + mask.unsqueeze(1)) / np.sqrt(512) #(batch, 1, seq_len)
        forward_summarized = torch.bmm(forward_out_score, forward_out).squeeze(1) #(batch, hidden)
        backward_summarized = torch.bmm(backward_out_score, backward_out).squeeze(1) #(batch, hidden)
        summarized = torch.cat([forward_summarized, backward_summarized], dim=1) #(batch, 2 * hidden)
        out = self.drop(self.act(self.hidden(summarized)))
        out = self.output(out)
        out = torch.reshape(out, (-1, 2, self.seq_len))
        start, end = out[:, 0], out[:, 1]
        return start, end

def convert_token_to_text(tokens):
    ret = ""
    for token in tokens:
        if token[0] in NOTATION:
            ret += token
        else:
            ret += ' ' + token
    return ret.strip()

def write_result(predicted_start, predicted_end, dev_example_list, info = ''):
    result_dict = dict()
    for i, example in enumerate(dev_example_list):
        stpos = predicted_start[i]
        edpos= predicted_end[i]
        if stpos >= edpos:
            result_dict[example.question_unique_id] = ""
        else:
            result_dict[example.question_unique_id] = convert_token_to_text(example.text_tokens[stpos: edpos + 1])
    with open("../result/BiGRU_Atttention_%s.json" % info, "w") as f:
        f.write(json.dumps(result_dict, indent=4) + "\n")

def main():
    epoch = 1
    #epoch = 0
    train_examples, dev_examples, word_to_index, embedding_matrix = load_data()
    vocab_size, embedding_size = embedding_matrix.shape
    train_dataset = recordDataset(train_examples)
    dev_dataset = recordDataset(dev_examples)
    model = BiGRUAttention(vocab_size, embedding_size, embedding_size, 512, embedding_matrix)
    train_loader = DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset), batch_size=48, num_workers=4)
    dev_loader = DataLoader(dataset=train_dataset, sampler=RandomSampler(dev_dataset), batch_size=48, num_workers=4)
    optimizer = Adam(model.parameters(), lr=1e-4)
    start_loss_function = nn.CrossEntropyLoss()
    end_loss_function = nn.CrossEntropyLoss()
    device = torch.device('cuda:1')
    model.to(device)
    model.train()
    def process_data(batch):
        (tokens, mask, start_label, end_label) = tuple(x.to(device) for x in batch)
        start, end = model(tokens, mask)
        start_loss = start_loss_function(start, start_label)
        end_loss = end_loss_function(end, end_label)
        loss = start_loss + end_loss
        return loss, start, end


    if epoch > 0:
        print("start training...")
        for e in range(epoch):
            print("EPOCH %d:" % e)
            cnt = 0.0
            total = 0.0
            for batch in tqdm.tqdm(train_loader):
                loss, start, end = process_data(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cnt += 1
                total += loss.item()
                # if (i + 1) % PRINT_EVERY == 0:
                #     print("EPOCH %d, BATCH %d, AVERAGE LOSS %.3f" % (e, i, float(total) / cnt))
            print("TRAIN LOSS OF EPOCH %d is %.5f" % (e, total / cnt))
            cnt = 0
            total = 0.0
            start_list = []
            end_list = []
            print("start evaluation...")
            for batch in tqdm.tqdm(dev_loader):
                with torch.no_grad():
                    loss, start, end = process_data(batch)
                    start_list.append(start.cpu())
                    end_list.append(end.cpu())
                    cnt += 1
                    total += loss.item()
            print("VALID LOSS OF EPOCH %d is %.5f" % (e, total / cnt))
            starts = np.concatenate(start_list, axis=0)
            ends = np.concatenate(end_list, axis=0)
            start_predicted = starts.argmax(axis=1)
            end_predicted  = ends.argmax(axis=1)
            write_result(start_predicted, end_predicted, dev_examples, "epoch_%d" % e)
            path = "../result/BiGRU-Attention_epoch%d.pkl"
            print("SAVE MODEL TO %s" % (path % e))
            torch.save(model.state_dict(), path % e)


    else:
        cnt = 0
        total = 0.0
        start_list = []
        end_list = []
        print("start evaluation...")
        for batch in tqdm.tqdm(dev_loader):
            with torch.no_grad():
                loss, start, end = process_data(batch)
                start_list.append(start.cpu())
                end_list.append(end.cpu())
                cnt += 1
                total += loss.item()
        print("VALID LOSS is %.5f" % (total / cnt))
        starts = np.concatenate(start_list, axis=0)
        ends = np.concatenate(end_list, axis=0)
        start_predicted = starts.argmax(axis=1)
        end_predicted = ends.argmax(axis=1)
        write_result(start_predicted, end_predicted, dev_examples, "final")
        path = "../result/BiGRU-Attention_final.pkl"
        print("SAVE MODEL TO %s" % (path))
        torch.save(model.state_dict(), path )


if __name__ == "__main__":
    # model = BiGRUAttention(300000, 300, 300, 512, np.random.randn(300000, 300))
    # tokens = torch.tensor(np.random.randint(0, 300000, (16, 2360)), dtype=torch.int64)
    # mask = torch.tensor(np.random.randint(0, 2, (16, 2360)), dtype=torch.float)
    # model(tokens, mask)
    main()