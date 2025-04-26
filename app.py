# app.py
from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import AutoTokenizer
from HateSpeechDetectionApp.preprocessing import preprocessing
import torch.nn.functional as F
import torch.nn as nn

# Định nghĩa lớp CNN
class CNN(nn.Module):
    # (giữ nguyên như trong app.py)
    # ...
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        self.conv_0 = nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_sizes[0])
        self.conv_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_sizes[1])
        self.conv_2 = nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_sizes[2])
        self.conv_3 = nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_sizes[3])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoded):
        embedded = self.fc_input(encoded)
        max_kernel_size = max(self.conv_0.kernel_size[0], self.conv_1.kernel_size[0], 
                              self.conv_2.kernel_size[0], self.conv_3.kernel_size[0])
        padding = max_kernel_size - 1
        embedded = F.pad(embedded, (0, 0, padding, 0))
        embedded = embedded.permute(0, 2, 1)
        
        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))
        conved_3 = F.relu(self.conv_3(embedded))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim=1))
        return self.fc(cat)

# Load model và tokenizer
model_phoBert_path = "module2_part1.pt"
model_cnn_path = "module2_part2.pt"
device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Cấu hình CNN
EMBEDDING_DIM = 768
N_FILTERS = 32
FILTER_SIZES = [1, 2, 3, 5]
OUTPUT_DIM = 3
DROPOUT = 0.1
PAD_IDX = tokenizer.pad_token_id

# Load model
cnn_model = CNN(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
phoBert = torch.load(model_phoBert_path, map_location=device, weights_only=False)
cnn_model=torch.load(model_cnn_path, map_location=device, weights_only=False)
phoBert.eval()
cnn_model.eval()

app = Flask(__name__)

label_mapping = {
    0: 2,   # Tích cực -> 2
    1: 1,   # Trung lập -> 1
    2: -1   # Tiêu cực -> -1
}

def predict(text):
    processed_text = preprocessing(text)
    inputs = tokenizer(processed_text, return_tensors="pt")
    
    with torch.no_grad():
        embedded = phoBert(inputs['input_ids'], inputs['attention_mask'])[0]
        predictions = cnn_model(embedded)
    
    predictions_np = predictions.detach().cpu().numpy()
    predicted_label = np.argmax(predictions_np, axis=1).flatten()[0]
    return label_mapping[predicted_label]

@app.route('/predict', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        # comment_id = data.get('comment_id')
        comment = data.get('comment')

        # if not comment_id or not comment:
        if not comment:
            return jsonify({"error": "Missing  'comment'"}), 400

        result = predict(comment)
        return jsonify({
            # "comment_id": comment_id,
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000,debug=False)
