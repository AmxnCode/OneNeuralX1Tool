# Google Colab Training Instructions

## Time Estimates
- **1 epoch:** ~3-5 minutes (T4 GPU)
- **2 epochs:** ~6-10 minutes
- **Total:** ~10-15 minutes (including setup)

## Step-by-Step Instructions

### 1. Upload to Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload Notebook**
3. Upload `colab_train.py` (or create a new notebook and paste cells)

### 2. Enable GPU
1. Click **Runtime → Change runtime type**
2. Select **GPU** (T4 recommended)
3. Click **Save**

### 3. Run All Cells
1. Click **Runtime → Run all**
2. Wait ~10-15 minutes

### 4. Push to HuggingFace
The script will ask for your HuggingFace token:
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (write access)
3. Paste token when prompted
4. Model uploads automatically to `YOUR_USERNAME/OneNeuralX1Tool-26M`

### 5. Download Model
The script will also download the model locally as a zip file.

## What the Script Does
1. **Installs dependencies** (tokenizers, huggingface_hub)
2. **Creates training data** (5000 multi-tool chain examples)
3. **Builds tokenizer** with 16384 vocab (tool names as single tokens)
4. **Trains 26M model** for 2 epochs
5. **Tests model** with sample queries
6. **Pushes to HuggingFace** (requires token)
7. **Downloads trained model** locally

## After Download
Extract `trained_model.zip` and use with:
```python
from your_model_code import OneNeuralX1ToolV2
import torch

model = OneNeuralX1ToolV2(vocab_size=16384)
model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
```

## Expected Results
- **Train loss:** ~0.05-0.1
- **Val accuracy:** ~80-85%
- **Test output:** Clean function calls like `get_weather`, `send_email`
