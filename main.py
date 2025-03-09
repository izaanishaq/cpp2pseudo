import streamlit as st
import torch
import math
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import base64


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Load local image
try:
    background_image = get_base64_encoded_image("image.jpeg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 1, .9), rgba(0, 0, 0, .9)), url("data:image/jpeg;base64,{background_image}");
            background-size: cover;
            background-attachment: fixed;
        }}
        header {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.warning(f"Unable to load background image: {e}")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(q, k.permute(0,1,3,2)) / self.scale.to(q.device)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.dropout(F.softmax(energy, dim=-1))
        output = torch.matmul(attention, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(output)
        return output

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, trg_mask, src_mask):
        self_attn_out = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        enc_attn_out = self.enc_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=256, num_heads=8,
                 num_layers=3, d_ff=512, dropout=0.1, max_len=128):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, trg_vocab_size)
        self.scale = math.sqrt(d_model)
        # Use BERT pad token id for both source and target (BERT tokenizer is used)
        self.src_pad_idx = BertTokenizer.from_pretrained('bert-base-uncased').pad_token_id
        self.trg_pad_idx = BertTokenizer.from_pretrained('bert-base-uncased').pad_token_id
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool().unsqueeze(0).unsqueeze(0)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        src = self.src_embedding(src) * self.scale
        src = src + self.pe[:, :src.size(1)].to(src.device)
        src = self.dropout(src)
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        trg = self.trg_embedding(trg) * self.scale
        trg = trg + self.pe[:, :trg.size(1)].to(trg.device)
        trg = self.dropout(trg)
        dec_output = trg
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, trg_mask, src_mask)
        output = self.output_layer(dec_output)
        return output

# --------- Helper Functions ---------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_path, tokenizer_info_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_info = {
        'src_vocab_size': checkpoint['src_vocab_size'],
        'trg_vocab_size': checkpoint['trg_vocab_size'],
        'max_length': checkpoint['max_length'],
        'd_model': checkpoint['d_model'],
        'num_heads': checkpoint['num_heads'],
        'num_layers': checkpoint['num_layers'],
        'd_ff': checkpoint['d_ff'],
        'dropout': checkpoint['dropout']
    }
    model = Transformer(
        src_vocab_size=model_info['src_vocab_size'],
        trg_vocab_size=model_info['trg_vocab_size'],
        d_model=model_info['d_model'],
        num_heads=model_info['num_heads'],
        num_layers=model_info['num_layers'],
        d_ff=model_info['d_ff'],
        dropout=model_info['dropout'],
        max_len=model_info['max_length']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
       
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 
    return model, bert_tokenizer

def tokenize_and_numericalize_test_bert(text, tokenizer, max_len=128):
    BOS_TOKEN = tokenizer.cls_token
    EOS_TOKEN = tokenizer.sep_token
    encoded = tokenizer.encode(
        BOS_TOKEN + " " + text + " " + EOS_TOKEN,
        add_special_tokens=False,
        max_length=max_len,
        truncation=True,
        padding='max_length'
    )
    return torch.tensor(encoded).unsqueeze(0)

def translate_code_to_pseudocode_bert(model, src_sequence, max_len, device, tokenizer):
    model.eval()
    src_sequence = src_sequence.to(device)
    src_mask = model.make_src_mask(src_sequence)
    with torch.no_grad():
        enc_output = model.forward(src_sequence, src_sequence)[:, :1, :]
    trg_indexes = [tokenizer.cls_token_id]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output = model.forward(src_sequence, trg_tensor)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == tokenizer.sep_token_id:
            break
    trg_tokens = tokenizer.convert_ids_to_tokens(trg_indexes)
    translated_text = tokenizer.convert_tokens_to_string(trg_tokens[1:-1])
    return translated_text

st.title("C++ to Pseudocode Translator")
st.write("Enter your C++ code below and click Translate.")


MODEL_SAVE_PATH = hf_hub_download(repo_id="izaanishaq/cpp-to-pseudocode", filename="cpp_to_pseudocode_model_bert_20epoch.pth")
TOKENIZER_INFO_PATH = hf_hub_download(repo_id="izaanishaq/cpp-to-pseudocode", filename="tokenizer_info_cpp_to_ps_bert_20epoch.txt")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with st.spinner("Loading model..."):
    model, tokenizer = load_model_and_tokenizer(MODEL_SAVE_PATH, TOKENIZER_INFO_PATH)
    model.to(device)

cpp_code_input = st.text_area("C++ Code Input", height=150)

if st.button("Translate"):
    if cpp_code_input.strip() == "":
        st.warning("Please enter some C++ code.")
    else:
        with st.spinner("Translating..."):
            max_len = 128
            numericalized = tokenize_and_numericalize_test_bert(cpp_code_input, tokenizer, max_len)
            output_text = translate_code_to_pseudocode_bert(model, numericalized, max_len, device, tokenizer)
        st.subheader("Predicted Pseudocode:")
        st.code(output_text, language="python")