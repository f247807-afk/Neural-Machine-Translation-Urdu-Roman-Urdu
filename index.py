
# =============================================================================
# DATA EXTRACTION MODULE
# =============================================================================
import zipfile
import os

zip_file_path = '/dataset.zip'
extract_dir = '/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extracted {zip_file_path} to {extract_dir}")

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================
 # Full cleaned pipeline: char-level transliteration (Urdu -> Roman)
import os, re, zipfile, glob, math, random
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

  # Optional BLEU
try:
      from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
      NLTK_AVAILABLE = True
except Exception:
      NLTK_AVAILABLE = False

  # -------------------------
  # Config / paths
  # -------------------------
DATA_DIR = "/content/dataset"  # adjust if needed


  # -------------------------
  # Utilities (dataset load + normalization)
  # -------------------------
# =============================================================================
# PREPROCESSING MODULE (DATA LOADING AND CLEANING)
# =============================================================================
def load_dataset(data_dir: str) -> pd.DataFrame:
      pairs = []
      for poet in os.listdir(data_dir):
          poet_dir = os.path.join(data_dir, poet)
          if not os.path.isdir(poet_dir):
              continue
          urdu_dir = os.path.join(poet_dir, "ur")
          roman_dir = os.path.join(poet_dir, "en")
          if not os.path.exists(urdu_dir) or not os.path.exists(roman_dir):
              continue
          for fname in os.listdir(urdu_dir):
              urdu_path = os.path.join(urdu_dir, fname)
              roman_path = os.path.join(roman_dir, fname)
              if os.path.isfile(urdu_path) and os.path.isfile(roman_path):
                  try:
                      with open(urdu_path, "r", encoding="utf-8") as f1, \
                          open(roman_path, "r", encoding="utf-8") as f2:
                          urdu_text = f1.read().strip()
                          roman_text = f2.read().strip()
                          if urdu_text and roman_text:
                              pairs.append((urdu_text, roman_text))
                  except Exception as e:
                      print(f"Skipping {fname} in {poet}: {e}")
      if not pairs:
          raise RuntimeError(f"No Urdu–Roman pairs found in {data_dir}")
      df = pd.DataFrame(pairs, columns=["urdu", "roman"])
      return df

def normalize_urdu(text: str) -> str:
      t = str(text).strip()
      t = t.replace("\u0640", "")
      t = t.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")
      t = re.sub(r'[\u064B-\u0652]', '', t)
      t = re.sub(r'\s+', ' ', t).strip()
      return t

_basic_map = {
      "ا":"a","آ":"aa","ب":"b","پ":"p","ت":"t","ث":"s","ج":"j","چ":"ch","ح":"h","خ":"kh",
      "د":"d","ذ":"z","ر":"r","ڑ":"r","ز":"z","ژ":"zh","س":"s","ش":"sh","ص":"s","ض":"z",
      "ط":"t","ظ":"z","ع":"'","غ":"gh","ف":"f","ق":"q","ک":"k","گ":"g","ل":"l","م":"m",
      "ن":"n","ں":"n","و":"o","ہ":"h","ھ":"h","ی":"y","ے":"e","ٔ":"'", "ُ":"u","ِ":"i","أ":"a"
}


def rule_based_transliterate(urdu_text: str) -> str:
      return "".join([_basic_map.get(ch, ch) for ch in urdu_text])


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['urdu'] = df['urdu'].astype(str).apply(normalize_urdu)
    df['roman'] = df['roman'].astype(str)

    print(f"Using all {len(df)} pairs (no length filtering)")
    print(f"Urdu length stats: min={df['urdu'].str.len().min()}, max={df['urdu'].str.len().max()}")
    print(f"Roman length stats: min={df['roman'].str.len().min()}, max={df['roman'].str.len().max()}")

    return df.reset_index(drop=True)


# =============================================================================
# VOCABULARY AND TOKENIZATION MODULE
# =============================================================================
# -------------------------
# Tokens / char vocab
# -------------------------
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

def build_char_vocab(sequences: List[str], min_freq: int = 1) -> Tuple[Dict[str,int], Dict[int,str]]:
    counter = Counter()
    for s in sequences:
        for ch in s:
            counter[ch] += 1

    print(f"Top 20 most common characters: {counter.most_common(20)}")
    print(f"Total unique characters: {len(counter)}")

    chars = [ch for ch, c in counter.items() if c >= min_freq]
    chars = SPECIAL_TOKENS + sorted(chars)

    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in itos.items()}

    return stoi, itos



def encode_sequence(seq: str, stoi: Dict[str,int], add_sos: bool=False, add_eos: bool=True) -> List[int]:
      ids = []
      if add_sos:
          ids.append(stoi[SOS_TOKEN])
      for ch in seq:
          ids.append(stoi.get(ch, stoi[UNK_TOKEN]))
      if add_eos:
          ids.append(stoi[EOS_TOKEN])
      return ids



# =============================================================================
# DATASET AND DATA LOADER MODULE
# =============================================================================    
# -------------------------
# Dataset + collate
# -------------------------
class TransliterationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str,str]], src_stoi, trg_stoi, max_src_len=200, max_trg_len=200):
        self.pairs = pairs
        self.src_stoi = src_stoi
        self.trg_stoi = trg_stoi
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_ids = encode_sequence(src, self.src_stoi, add_sos=False, add_eos=True)[:self.max_src_len]
        trg_ids = encode_sequence(trg, self.trg_stoi, add_sos=True, add_eos=True)[:self.max_trg_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def make_collate_fn(src_stoi, trg_stoi):
    PAD_SRC = src_stoi[PAD_TOKEN]
    PAD_TRG = trg_stoi[PAD_TOKEN]

    def collate_fn(batch):
        srcs, trgs = zip(*batch)
        src_lens = torch.tensor([s.size(0) for s in srcs], dtype=torch.long)
        trg_lens = torch.tensor([t.size(0) for t in trgs], dtype=torch.long)
        src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD_SRC)
        trg_pad = nn.utils.rnn.pad_sequence(trgs, batch_first=True, padding_value=PAD_TRG)
        return src_pad, src_lens, trg_pad, trg_lens

    return collate_fn


def build_dataloaders_char(df, src_stoi, trg_stoi, cfg, batch_size=32):
    pairs = list(zip(df['urdu'].tolist(), df['roman'].tolist()))
    ds = TransliterationDataset(
        pairs, src_stoi, trg_stoi,
        max_src_len=cfg.get("max_src_len", 200),
        max_trg_len=cfg.get("max_trg_len", 200)
    )

    total = len(ds)

    # Handle small datasets - ensure minimum sizes
    if total < 10:
        # For very small datasets, use 70%/15%/15% split
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
    else:
        train_size = int(0.5 * total)
        val_size = int(0.25 * total)
        test_size = total - train_size - val_size

    # Ensure minimum sizes of at least 1
    train_size = max(1, train_size)
    val_size = max(1, val_size)
    test_size = max(1, test_size)

    # Adjust if total is too small
    if train_size + val_size + test_size > total:
        test_size = max(1, total - train_size - val_size)

    train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])

    collate = make_collate_fn(src_stoi, trg_stoi)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, train_size), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_size), collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, test_size), collate_fn=collate)

    print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    return train_loader, val_loader, test_loader



# =============================================================================
# MODEL ARCHITECTURE MODULES
# =============================================================================
# ------------------------------
# Encoder: BiLSTM
# ------------------------------
class EncoderBiLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hidden_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            enc_hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens=None):
        embedded = self.dropout(self.embedding(src))
        if src_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)  




# ------------------------------
# Attention
# ------------------------------
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim*2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) 
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  
        return F.softmax(attention, dim=1)

# ------------------------------
# Decoder: LSTM with Attention
# ------------------------------
class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers=4, dropout=0.5, attention=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(
            (enc_hidden_dim*2)+emb_dim,
            dec_hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear((enc_hidden_dim*2)+dec_hidden_dim+emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_proj = nn.Linear(enc_hidden_dim * 2, self.rnn.hidden_size)
        self.cell_proj = nn.Linear(enc_hidden_dim * 2, self.rnn.hidden_size)
    def init_hidden(self, enc_hidden, enc_cell):
        hidden_cat = torch.cat((enc_hidden[-2], enc_hidden[-1]), dim=1)
        cell_cat = torch.cat((enc_cell[-2], enc_cell[-1]), dim=1)

        hidden = torch.tanh(self.hidden_proj(hidden_cat)).unsqueeze(0)
        cell = torch.tanh(self.cell_proj(cell_cat)).unsqueeze(0)
        return hidden, cell

    def forward(self, input, hidden, cell, encoder_outputs):
      
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))  

        a = self.attention(hidden[-1], encoder_outputs) 
        a = a.unsqueeze(1)  
        weighted = torch.bmm(a, encoder_outputs)  

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2).squeeze(1))
        return prediction, hidden, cell, a.squeeze(1)


# ------------------------------
# Seq2Seq: Wrapper
# ------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx

        self.hidden_proj = nn.Linear(encoder.enc_hidden_dim*2, decoder.rnn.hidden_size)
        self.cell_proj   = nn.Linear(encoder.enc_hidden_dim*2, decoder.rnn.hidden_size)
    def init_decoder_hidden(self, enc_hidden, enc_cell):
        hidden_cat = torch.cat((enc_hidden[-2], enc_hidden[-1]), dim=1)
        cell_cat   = torch.cat((enc_cell[-2], enc_cell[-1]), dim=1)
        dec_hidden = torch.tanh(self.hidden_proj(hidden_cat)).unsqueeze(0)
        dec_cell   = torch.tanh(self.cell_proj(cell_cat)).unsqueeze(0)
        return dec_hidden, dec_cell

    def forward(self, src, src_lens, trg=None, teacher_forcing_ratio=0.5, max_trg_len=100):
         # 1. Encoder forward
        encoder_outputs, (enc_hidden, enc_cell) = self.encoder(src, src_lens)

    # 2. Combine forward & backward states from **last BiLSTM layer**
    
        hidden_cat = torch.cat((enc_hidden[-2], enc_hidden[-1]), dim=1)
        cell_cat   = torch.cat((enc_cell[-2], enc_cell[-1]), dim=1)


        dec_hidden = torch.tanh(self.hidden_proj(hidden_cat)).unsqueeze(0)  
        dec_cell   = torch.tanh(self.cell_proj(cell_cat)).unsqueeze(0)
        
        dec_hidden = dec_hidden.repeat(self.decoder.rnn.num_layers, 1, 1)
        dec_cell   = dec_cell.repeat(self.decoder.rnn.num_layers, 1, 1)


        trg_len = trg.size(1) if trg is not None else max_trg_len
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(self.device)

        input = trg[:,0] if trg is not None else torch.full((batch_size,), self.sos_idx, dtype=torch.long).to(self.device)

        for t in range(1, trg_len):
            output, dec_hidden, dec_cell, _ = self.decoder(input, dec_hidden, dec_cell, encoder_outputs)
            outputs[:,t,:] = output
            teacher_force = trg is not None and torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:,t] if teacher_force else output.argmax(1)

        return outputs


def build_model(cfg, src_stoi, trg_stoi, device):
    input_dim = len(src_stoi)
    output_dim = len(trg_stoi)

    enc = EncoderBiLSTM(
        input_dim,
        cfg["emb_dim"],
        cfg["enc_hidden"],
        n_layers=cfg["enc_layers"],
        dropout=cfg["dropout"]
    )

    attn = Attention(cfg["enc_hidden"], cfg["enc_hidden"])
    dec = DecoderLSTM(
        output_dim,
        cfg["emb_dim"],
        cfg["enc_hidden"],
        cfg["enc_hidden"], 
        n_layers=cfg["dec_layers"],
        dropout=cfg["dropout"],
        attention=attn
    )

    model = Seq2Seq(enc, dec, device, sos_idx=trg_stoi['<SOS>']).to(device)
    return model



# =============================================================================
# EVALUATION METRICS MODULE
# =============================================================================
def levenshtein(a: str, b: str) -> int:
      n, m = len(a), len(b)
      if n == 0: return m
      if m == 0: return n
      dp = [[0]*(m+1) for _ in range(n+1)]
      for i in range(n+1): dp[i][0] = i
      for j in range(m+1): dp[0][j] = j
      for i in range(1, n+1):
          for j in range(1, m+1):
              cost = 0 if a[i-1] == b[j-1] else 1
              dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
      return dp[n][m]

def cer(reference: str, hypothesis: str) -> float:
      d = levenshtein(reference, hypothesis)
      if len(reference) == 0: return 1.0 if len(hypothesis)>0 else 0.0
      return d / len(reference)

def compute_bleu(reference: str, hypothesis: str) -> float:
      if NLTK_AVAILABLE:
          smoothie = SmoothingFunction().method1
          return sentence_bleu([list(reference)], list(hypothesis), smoothing_function=smoothie)
      else:
          ref = list(reference); hyp = list(hypothesis)
          ref_counts = Counter(ref)
          match = 0
          for t in hyp:
              if ref_counts.get(t,0) > 0:
                  match += 1
                  ref_counts[t] -= 1
          return match / max(1, len(hyp))

# =============================================================================
# TRAINING MODULE
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, clip=5.0, epoch=None, total_epochs=None,
                teacher_forcing_start=0.9, teacher_forcing_end=0.3):
    
      if epoch is not None and total_epochs is not None:
        progress = epoch / total_epochs
        teacher_forcing_ratio = teacher_forcing_start + (teacher_forcing_end - teacher_forcing_start) * progress
      else:
        teacher_forcing_ratio = 0.5

      model.train()
      total_loss = 0.0

      for src, src_lens, trg, trg_lens in tqdm(dataloader, desc="Train batches"):
        src, trg = src.to(device), trg.to(device)
        src_lens, trg_lens = src_lens.to(device), trg_lens.to(device)

        optimizer.zero_grad()
        outputs = model(src, src_lens,trg, teacher_forcing_ratio)
        output_dim = outputs.size(-1)
        loss = criterion(outputs.view(-1, output_dim), trg.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

      return total_loss / len(dataloader)

# =============================================================================
# EVALUATION MODULE
# =============================================================================
def evaluate(model, dataloader, criterion, device, trg_itos):
      model.eval()  
      total_loss = 0.0
      total_cer = 0.0
      total_bleu = 0.0
      n_sent = 0

      with torch.no_grad():  
        for src, src_lens, trg, trg_lens in tqdm(dataloader, desc="Eval batches"):
            src, trg = src.to(device), trg.to(device)
            src_lens, trg_lens = src_lens.to(device), trg_lens.to(device)

            outputs = model(src, src_lens, trg=None, teacher_forcing_ratio=0.0, max_trg_len=trg.size(1))

            # Calculate loss 
            loss = criterion(outputs.view(-1, outputs.size(-1)), trg.view(-1))
            total_loss += loss.item()

            # Decode predictions and references
            preds = outputs.argmax(-1).cpu().numpy()
            trg_np = trg.cpu().numpy()

            for i in range(preds.shape[0]):
                def decode_idx_seq(idx_seq):
                    chars = []
                    for idx in idx_seq:
                        ch = trg_itos.get(int(idx), UNK_TOKEN)
                        if ch == EOS_TOKEN:
                            break
                        if ch in SPECIAL_TOKENS:
                            continue
                        chars.append(ch)
                    return "".join(chars)

                pred_s = decode_idx_seq(preds[i])
                ref_s = decode_idx_seq(trg_np[i])
                total_cer += cer(ref_s, pred_s)
                total_bleu += compute_bleu(ref_s, pred_s)
                n_sent += 1

      avg_loss = total_loss / len(dataloader)
      avg_cer = total_cer / max(1, n_sent)
      avg_bleu = total_bleu / max(1, n_sent)
      perp = math.exp(avg_loss) if avg_loss < 100 else float('inf')

      return {"loss": avg_loss, "perplexity": perp, "CER": avg_cer, "BLEU": avg_bleu}


# ===============
# beam search
# ===============
def translate_beam_search(model, src_text: str, src_stoi, trg_itos, trg_stoi, device,
                          max_len=200, beam_width=3, ngram_block=3):
    """
    Beam search decoding for seq2seq model with length normalization and repetition handling.
    """
    model.eval()
    src_ids = encode_sequence(normalize_urdu(src_text), src_stoi, add_sos=False, add_eos=True)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(src_ids)]).to(device)

    with torch.no_grad():
        encoder_outputs, (enc_hidden, enc_cell) = model.encoder(src_tensor, src_len)

        # Initialize decoder hidden/cell
        hidden_cat = torch.cat((enc_hidden[-2], enc_hidden[-1]), dim=1)
        cell_cat   = torch.cat((enc_cell[-2], enc_cell[-1]), dim=1)

        dec_hidden = torch.tanh(model.hidden_proj(hidden_cat)).unsqueeze(0).repeat(
            model.decoder.rnn.num_layers, 1, 1
        )
        dec_cell   = torch.tanh(model.cell_proj(cell_cat)).unsqueeze(0).repeat(
            model.decoder.rnn.num_layers, 1, 1
        )

        SOS_IDX = trg_stoi[SOS_TOKEN]
        EOS_IDX = trg_stoi[EOS_TOKEN]

        beams = [(torch.tensor([SOS_IDX], device=device), dec_hidden, dec_cell, 0.0)]

        # Helper function for n-gram blocking
        def has_repeat_ngram(toks, n=3):
            if len(toks) < n:
                return False
            ngrams = [tuple(toks[j:j+n].tolist()) for j in range(len(toks)-n+1)]
            return len(ngrams) != len(set(ngrams))

        for _ in range(max_len):
            new_beams = []
            for seq, hidden, cell, score in beams:
                input_token = seq[-1].unsqueeze(0)
                output, hidden_new, cell_new, attn_weights = model.decoder(
                    input_token, hidden, cell, encoder_outputs
                )

                log_probs = torch.log_softmax(output, dim=1).squeeze(0) 
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_token = topk_indices[i].unsqueeze(0)
                    next_score = score + topk_probs[i].item()

                    # Repetition penalty
                    if next_token.item() in seq.tolist():
                        next_score -= 1.0

                    new_seq = torch.cat([seq, next_token])

                    if ngram_block and has_repeat_ngram(new_seq, n=ngram_block):
                        continue

                    # Length normalization
                    length_norm = (5 + len(new_seq)) ** 0.7 / (5 + 1) ** 0.7
                    next_score = next_score / length_norm

                    new_beams.append((new_seq, hidden_new, cell_new, next_score))

            if not new_beams:
              
                new_beams = beams

            beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_width]

            if all(seq[-1].item() == EOS_IDX for seq, _, _, _ in beams):
                break

        best_seq = beams[0][0]

        # Decode tokens to chars
        chars = []
        for idx in best_seq[1:]: 
            ch = trg_itos.get(int(idx.item()), UNK_TOKEN)
            if ch == EOS_TOKEN:
                break
            if ch in SPECIAL_TOKENS:
                continue
            chars.append(ch)

    return "".join(chars)



# =============================================================================
# MAIN PIPELINE MODULE
# =============================================================================
def run_pipeline(data_dir, experiment_configs, device_str="cuda"):
      device = torch.device(device_str if torch.cuda.is_available() else "cpu")
      print("Using device:", device)
      df = load_dataset(data_dir)
      df = prepare_dataframe(df)
      print(f"Loaded {len(df)} pairs after cleaning.")

    # character vocabs
      src_stoi, src_itos = build_char_vocab(df['urdu'].tolist())
      trg_stoi, trg_itos = build_char_vocab(df['roman'].tolist())
      print("Vocab sizes -> src:", len(src_stoi), "trg:", len(trg_stoi))

      results = []
      for cfg in experiment_configs:
        print("Experiment:", cfg)
        train_loader, val_loader, test_loader = build_dataloaders_char(df, src_stoi, trg_stoi, cfg, batch_size=cfg.get("batch_size",8))
        model = build_model(cfg, src_stoi, trg_stoi, device=device)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
      
        pad_idx = trg_stoi[PAD_TOKEN]
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        best_val = float('inf')
        patience = cfg.get("patience", 3) 
        patience_counter = 0
        best_state = None

        for epoch in range(cfg["epochs"]):
           
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip=1.0, epoch=epoch,total_epochs=cfg["epochs"])

            scheduler.step()
           
            model.eval()
            val_metrics = evaluate(model, val_loader, criterion, device, trg_itos)

            scheduler.step(val_metrics['loss'])

            print(f"Epoch {epoch+1}/{cfg['epochs']} TrainLoss: {train_loss:.4f} ValLoss: {val_metrics['loss']:.4f} BLEU: {val_metrics['BLEU']:.4f} CER: {val_metrics['CER']:.4f}")

            # Early stopping
            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

        model.eval()
        test_metrics = evaluate(model, test_loader, criterion, device, trg_itos)
        print("Test:", test_metrics)
        results.append({"cfg": cfg, "model": model, "src_stoi": src_stoi, "src_itos": src_itos, "trg_stoi": trg_stoi, "trg_itos": trg_itos, "test_metrics": test_metrics})

      return results



# =============================================================================
# EXPERIMENT CONFIGURATIONS AND EXECUTION
# =============================================================================
# Example experiment configuration

experiments = [
    {
        "emb_dim": 128,          # Embedding dimension
        "enc_hidden": 256,       # Hidden size of BiLSTM encoder
        "enc_layers": 2,         # Number of BiLSTM encoder layers
        "dec_hidden": 256,       # Decoder hidden size (optional, can match encoder)
        "dec_layers": 2,         # Number of decoder LSTM layers
        "dropout": 0.3,          # Dropout rate
        "lr": 5e-4,              # Learning rate
        "batch_size": 64,        # Batch size
        "epochs": 1,
        "patience": 5,
        "max_src_len": 100,
        "max_trg_len": 120,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "teacher_forcing_start": 0.9,
        "teacher_forcing_end": 0.3
    },
    {
        "emb_dim": 256,
        "enc_hidden": 512,
        "enc_layers": 3,
        "dec_hidden": 512,
        "dec_layers": 3,
        "dropout": 0.5,
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "patience": 5,
        "max_src_len": 100,
        "max_trg_len": 120,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "teacher_forcing_start": 0.9,
        "teacher_forcing_end": 0.3
    },
    {
        "emb_dim": 512,
        "enc_hidden": 512,
        "enc_layers": 4,
        "dec_hidden": 512,
        "dec_layers": 4,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 1,
        "patience": 5,
        "max_src_len": 100,
        "max_trg_len": 120,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "teacher_forcing_start": 0.9,
        "teacher_forcing_end": 0.3
    }
]

# =============================================================================
# DEMONSTRATION AND VISUALIZATION MODULE
# =============================================================================
def demo_transliteration(model, src_stoi, trg_itos, device, examples):
    """Demonstrate transliteration on example sentences"""
    model.eval()
    for urdu_text, expected_roman in examples:
        print(f"Urdu: {urdu_text}")
        print(f"Expected: {expected_roman}")
        pred_beam = translate_beam_search(model, urdu_text, src_stoi, trg_itos, trg_stoi, device)
        print(f"Predicted (beam): {pred_beam}")
        print(f"CER: {cer(expected_roman, pred_beam):.4f}")
        print("-" * 50)

# Example demonstration
if results:
    best_result = results[0]  # Get the first result
    demo_examples = [
        ("ہیلو دنیا", "hello duniya"),
        ("کیا حال ہے", "kya haal hai"),
        ("میں ٹھیک ہوں", "main theek hoon")
    ]
    demo_transliteration(best_result["model"], best_result["src_stoi"], best_result["trg_itos"], torch.device("cuda" if torch.cuda.is_available() else "cpu"), demo_examples)


