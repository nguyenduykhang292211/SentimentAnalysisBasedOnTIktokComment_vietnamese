from Pineline_preprocessing import*
import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer

MODEL_NAME='vinai/phobert-base-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 40
class_names=['negative','neutral','positive']
# Set GPU
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device = torch.device(device)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.phobert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits
    
model = SentimentClassifier(3)
model.load_state_dict(torch.load('Model/train_3.bin'))
model.to(device)

def encode_predict(comment_test):
    encoded_review = tokenizer.encode_plus(
        comment_test,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    
    return class_names[prediction]

def predict(input_file, output_file):
    df=preprocess_without_output(input_file)
    df['Label']=df['Comment'].apply(encode_predict)
    df.to_csv(output_file,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Vietnamese text data.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    predict(args.input_file, args.output_file)