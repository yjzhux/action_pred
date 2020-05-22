"""LSTM models used in this project. """

import torch
import torch.nn as nn
import ipdb

class LSTM(nn.Module):
    '''Basic LSTM model takes a video sequence as input and one frame each timestep.
    '''
    def __init__(self, input_size, hidden_dim, n_class, n_layers=1):
        super(LSTM,self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, n_class)

    
    def forward(self,x):
        '''x shape (batch, time_step, input_size)
        r_out shape (batch, time_step, output_size)
        There are two hidden states in LSTM, namely h_n, h_c:
            h_n shape (n_layers, batch, hidden_size)   
            h_c shape (n_layers, batch, hidden_size)
        '''
        # None represents hidden state with zero initialization
        r_out, (h_n, h_c) = self.lstm(x, None) 
        model_out = self.hidden2out(r_out)
        # ipdb.set_trace()
        # return outdat
        return model_out, r_out


class GRU(nn.Module):
    '''Basic GRU model takes a video sequence as input and one frame each timestep.
    '''
    def __init__(self, input_size, hid_dim, n_class, n_layers=1):
        super(GRU,self).__init__()
        self.input_dim = input_size
        self.hid_dim = hid_dim
        self.gru = nn.GRU(input_size, hid_dim, n_layers, batch_first=True)
        self.h2class = nn.Linear(hid_dim, n_class)

    
    def forward(self,x):
        '''x shape (batch, time_step, input_size)
        hidden shape (batch, time_step, output_size)
        h_n shape (n_layers, batch, hidden_size)   
        '''
        # None represents hidden state with zero initialization
        hidden, h_n = self.gru(x, None) 
        class_out = self.h2class(hidden)
        # ipdb.set_trace()
        # return outdat
        return class_out, hidden


class Seq2Seq(nn.Module):
    """Combine encoder and decoder to form a seq2seq model. However, this model is
    different from traditional one because the encoder is an independent one. The 
    encoder is firstly trained then each of its hidden states is fed into the decoder
    for training. 
    """
    def __init__(self, encoder, decoder, f_length, task='joint'):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.f_length = f_length
        self.n_class = decoder.n_class
        self.n_pose = decoder.n_pose
        # task: class, pose, joint
        self.task = task
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, x):
        # x.shape = (batch_size, seq_len, feat_dim), where batch_size=1
        # batch_first=False
        # en_out.shape = (batch_size, ..., hid_dim)
        # en_class.shape = (batch_size, ..., n_class)
        # en_pose.shape = (batch_size, ..., n_pose)
        seq_len = x.size(1)
        # use pose_gt as the input of decoder
        en_pose = x[:, :, -self.n_pose:][:]
        en_class, en_hidden = self.encoder(x)

        # Take the outputs of encoder as the inputs of decoder

        # ----- the first timestep of decoder -----
        class_outs = torch.zeros(self.f_length, seq_len, self.n_class).cuda()
        class_outs[0][:] = en_class[0][:]
        pose_outs = torch.zeros(self.f_length, seq_len, self.n_pose).cuda()
        pose_outs[0][:] = en_pose[0][:]
        # de_input.shape = (seq_len, batch, input_size), where seq_len = 1
        # de_hidden.shape = (n_layers, batch, hid_dim), where n_layers = 1
        if self.task == 'class':
            de_input = en_class[0][:].unsqueeze(0)
        elif self.task == 'pose':
            de_input = en_pose[0][:].unsqueeze(0)
        elif self.task == 'joint':
            de_input = torch.cat((en_class, en_pose), 2)[0][:].unsqueeze(0)
        else:
            raise Exception("Unsupport task!", self.task)
        de_hidden = en_hidden[0][:].unsqueeze(0)

        # ----- the following timesteps of decoder -----
        # Training without teacher forcing: use its own predictions as the next input
        # ipdb.set_trace()
        for t in range(1, self.f_length):
            de_class, de_pose, de_hidden = self.decoder(de_input, de_hidden)
            # collect prediction results of decoder
            class_outs[t] = de_class[0][:]
            pose_outs[t] = de_pose[0][:]
            # update inputs
            if self.task == 'class':
                de_input = de_class
            elif self.task == 'pose':
                de_input = de_pose
            elif self.task == 'joint':
                de_input = torch.cat((de_class, de_pose), 2)
            else:
                raise Exception("Unsupport task!", self.task)
            
        return class_outs, pose_outs


class Encoder(nn.Module):
    '''Use GRU as Encoder because it has only one hidden state that can be preserved every 
    timestep. LSTM has two hidden states, i.e., hidden state and cell state. The cellstate 
    cannot be passed to Decoder because only the last cell state is outputed.  
    '''
    def __init__(self, input_size, hid_dim, n_class, n_layers):
        super(Encoder,self).__init__()
        self.input_dim = input_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hid_dim, n_layers, batch_first=True)
        self.h2class = nn.Linear(hid_dim, n_class)
    
    def forward(self,x):
        # None represents hidden state with zero initialization
        # hidden.shape = (seq_len, batch_size, hid_dim), where batch_size = 1
        # h_n.shape = (n_layers, batch_size, hid_dim), where n_layers = 1
        hidden, h_n = self.gru(x, None) 
        class_out = self.h2class(hidden)
        # pose_out = self.h2pose(hidden)
        # return class_out, h_n
        return class_out, hidden


class Decoder(nn.Module):
    '''Use GRU as Decoder to predict future labels. Original decoder.
    '''
    def __init__(self, input_size, hid_dim, n_class, n_layers, n_pose):
        super(Decoder,self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.n_pose = n_pose
        # self.gru = nn.GRU(input_size + n_pose, hid_dim, n_layers)
        self.gru = nn.GRU(input_size, hid_dim, n_layers)
        self.h2class = nn.Linear(hid_dim, n_class)
        self.h2pose = nn.Linear(hid_dim, n_pose)

    
    def forward(self, x, hidden):
        # None represents hidden state with zero initialization
        # ipdb.set_trace()
        # the inputs of the decoder x.shape = (seq_len, batch_size, feat_dim)
        hidden, h_n = self.gru(x, None) 
        class_out = self.h2class(hidden)
        pose_out = self.h2pose(hidden)
        return class_out, pose_out, hidden


class Decoder0(nn.Module):
    '''Add a hidden layer to the linear module.
    '''
    def __init__(self, input_size, hid_dim, n_class, n_layers, n_pose):
        super(Decoder0,self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.n_pose = n_pose
        # self.gru = nn.GRU(input_size + n_pose, hid_dim, n_layers)
        self.gru = nn.GRU(input_size, hid_dim, n_layers)
        self.h2class = nn.Sequential(nn.Linear(hid_dim, n_pose), nn.Linear(n_pose, n_class))
        self.h2pose = nn.Sequential(nn.Linear(hid_dim, n_class), nn.Linear(n_class, n_pose))

    
    def forward(self, x, hidden):
        # None represents hidden state with zero initialization
        # ipdb.set_trace()
        # the inputs of the decoder x.shape = (seq_len, batch_size, feat_dim)
        hidden, h_n = self.gru(x, None) 
        class_out = self.h2class(hidden)
        pose_out = self.h2pose(hidden)
        return class_out, pose_out, hidden


class Decoder0_1(nn.Module):
    '''Add hidden outputs.
    '''
    def __init__(self, input_size, hid_dim, n_class, n_layers, n_pose):
        super(Decoder0_1,self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.n_pose = n_pose
        # self.gru = nn.GRU(input_size + n_pose, hid_dim, n_layers)
        self.gru = nn.GRU(input_size, hid_dim, n_layers)
        # hidden --> class hidden --> pose
        self.h2ch = nn.Linear(hid_dim, n_class)
        self.ch2p = nn.Linear(n_class, n_pose)
        # hidden --> pose hidden --> class
        self.h2ph = nn.Linear(hid_dim, n_pose)
        self.ph2c = nn.Linear(n_pose, n_class)

    
    def forward(self, x, hidden):
        # None represents hidden state with zero initialization
        # ipdb.set_trace()
        # the inputs of the decoder x.shape = (seq_len, batch_size, feat_dim)
        hidden, h_n = self.gru(x, None) 

        ph = self.h2ph(hidden)
        class_out = self.ph2c(ph)

        ch = self.h2ch(hidden)
        pose_out = self.ch2p(ch)

        return class_out, pose_out, hidden
        # return (class_out + ch)/2, (pose_out + ph)/2, hidden


class Decoder1(nn.Module):
    '''Use GRU as Decoder to predict future labels. 
    Version 1: the initial hidden state, $h_0$ is the context vector, $z$.

    Linear: $$\hat{y}_{t+1} = f(h_t)$$
    GRU: $$h_t = \text{DecoderGRU}(y_t, h_{t-1})$$
    '''
    def __init__(self, input_size, hid_dim, n_class, n_layers, n_pose):
        self.input_dim = input_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.n_pose = n_pose
        self.gru = nn.GRU(input_size + n_pose, hid_dim, n_layers)
        self.h2class = nn.Linear(hid_dim, n_class)
        self.h2pose = nn.Linear(hid_dim, n_pose)

    
    def forward(self, x, h_n, context):
        # x.shape = [batch_size, ]
        # h_n.shape = [n_layers * n_directions, batch size, hid_dim]
        # n_layers and n_directions in the decoder will both be 1, then:
        # h_n.shape = [1, batch_size, hid_dim]
        # context is the vector $z$ from encoder
        
        hidden, h_n = self.gru(x, h_n) 
        class_out = self.h2class(hidden)
        pose_out = self.h2pose(hidden)
        return class_out, pose_out, h_n


class Decoder2(nn.Module):
    '''Add one input to linear layer (hidden to class/pose).
    Version 2: Instead of taking just the hidden state $h_n$ as input, the linear
    layer also takes the current target token, $y_t$. 

    Linear: $\hat{y}_{t+1} = f(h_t, y_t)$
    GRU (sanme with ver.1): $h_t = \text{DecoderGRU}(y_t, h_{t-1})$
    '''


class Decoder3(nn.Module):
    '''Add one input (the context vector from decoder, $z$) to every GRU unit.
    Version 3: Instead of taking just the target token, $y_t$ and the previous 
    hidden state $h_{t-1}$ as inputs, it also takes the context vector $z$. It 
    means we re-use the same context vector returned by the encoder for every 
    time-step in the decoder. 

    Note, the initial hidden state, $s_0$, is still the context vector, $z$, so 
    when generating the first token we are actually inputting two identical 
    context vectors into the GRU.

    Linear (sanme with ver.1): $\hat{y}_{t+1} = f(h_t)$
    GRU: $h_t = \text{DecoderGRU}(y_t, h_{t-1}, z)$
    '''


class Decoder4(nn.Module):
    '''Combine ver.2 and ver.3

    Linear : $\hat{y}_{t+1} = f(h_t, y_t)$
    GRU: $h_t = \text{DecoderGRU}(y_t, h_{t-1}, z)$
    '''


class Decoder5(nn.Module):
    '''In addition to ver.4 (combining ver.2 and ver.3), add the context vector
    $z$ as an additional input of the linear layer.

    Linear : $\hat{y}_{t+1} = f(h_t, y_t, z)$
    GRU: $h_t = \text{DecoderGRU}(y_t, h_{t-1}, z)$
    '''



class LSTMclsReg(nn.Module):
    '''Basic LSTM model takes a video sequence as input and one frame each timestep.
    It predict one future label as well as the corresponding pose (8 * 2 = 16). So 
    there are two tasks: classification and regression. They are learned jointly.
    '''
    def __init__(self, input_size, hidden_dim, n_class, n_layers=1, f_length=5, n_pose=16):
        super(LSTMclsReg,self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.h2class = nn.Linear(hidden_dim, n_class)
        self.h2pose = nn.Linear(hidden_dim, n_pose)

    
    def forward(self,x):
        '''x shape (batch, time_step, input_size)
        r_out shape (batch, time_step, output_size)
        '''
        # None represents hidden state with zero initialization
        r_out, _ = self.lstm(x, None) 
        class_out = self.h2class(r_out)
        pose_out = self.h2pose(r_out)
        # ipdb.set_trace()
        # return outdat
        return class_out, pose_out


class LSTMmultiLabel(nn.Module):
    '''Basic LSTM model takes a video sequence as input and one frame each timestep.
    It treats future labels as a multi-label classification problem. Onre disacvantage 
    of this model is that the future length is fixed in the model. 
    '''
    def __init__(self, input_size, hidden_dim, n_class, n_layers=1, f_length=5):
        super(LSTMmultiLabel,self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # self.hidden2out = nn.Linear(hidden_dim, n_class)
        # each head predicts one label
        self.hidden2out = nn.ModuleList()
        for idx in range(f_length):
            self.hidden2out.append(nn.Linear(hidden_dim, n_class))

    
    def forward(self,x):
        '''x shape (batch, time_step, input_size)
        r_out shape (batch, time_step, output_size)
        There are two hidden states in LSTM, namely h_n, h_c:
            h_n shape (n_layers, batch, hidden_size)   
            h_c shape (n_layers, batch, hidden_size)
        '''
        # None represents hidden state with zero initialization
        # ipdb.set_trace()
        r_out, (h_n, h_c) = self.lstm(x, None) 
        # model_out is a list with multi putpurts
        model_out = [fc(r_out) for fc in self.hidden2out]
        
        return model_out, r_out

