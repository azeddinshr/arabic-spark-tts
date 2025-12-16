# Spark-TTS Arabic Fine-tuning

Fine-tuning [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) on Arabic dataset (ClArTTS - 12 hours of Classical Arabic speech).

## 📦 Repository Contents

- Training pipeline for Arabic TTS fine-tuning
- Data preparation scripts
- Training configuration files
- Inference examples

**📥 Pre-trained Model:** [azeddinShr/Spark-TTS-Arabic-Complete](https://huggingface.co/azeddinShr/Spark-TTS-Arabic-Complete)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone required repositories
git clone https://github.com/SparkAudio/Spark-TTS
git clone https://github.com/ductran150499/SparkTTS-Finetune

# Install dependencies
pip install transformers soundfile huggingface_hub omegaconf torch datasets librosa
```

### 2. Data Preparation

```python
from datasets import load_dataset
import soundfile as sf
import librosa
import numpy as np
import os

# Load ClArTTS dataset
dataset = load_dataset("MBZUAI/ClArTTS")

# Create output directory
os.makedirs("data", exist_ok=True)

# Process and save audio + text pairs
for idx, sample in enumerate(dataset['train']):
    # Get audio and resample to 24kHz
    audio = np.array(sample['audio'])
    audio_24k = librosa.resample(audio, orig_sr=sample['sampling_rate'], target_sr=24000)
    text = sample['text']
    
    # Save audio file
    sf.write(f"data/{idx:05d}.wav", audio_24k, 24000)
    
    # Save corresponding text file
    with open(f"data/{idx:05d}.txt", 'w', encoding='utf-8') as f:
        f.write(text)

print(f"✅ Processed {len(dataset['train'])} samples")
```

### 3. Create Metadata

```bash
cd SparkTTS-Finetune
python create_metadata.py
```

This creates `metadata.csv` with format: `audio_path,text`

### 4. Extract Semantic Tokens

```bash
python -m src.process_data \
  --data_dir /path/to/data \
  --output_dir /path/to/processed_output
```

This extracts semantic tokens from audio using BiCodec, creating training pairs for the LLM.

### 5. Install Training Framework

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
```

### 6. Configure Training

Edit `config_axolotl/full_finetune.yml`:

```yaml
base_model: /path/to/Spark-TTS-0.5B/LLM
datasets:
  - path: /path/to/processed_output/data.jsonl
    type: completion
output_dir: /path/to/output
num_epochs: 20
micro_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002
```

### 7. Train

```bash
cd SparkTTS-Finetune
accelerate launch -m axolotl.cli.train config_axolotl/full_finetune.yml
```

Training time: ~3-4 hours on single GPU for 30% of dataset

---

## 🎤 Inference

### Using Pre-trained Model

```python
from huggingface_hub import snapshot_download
import sys
import torch
import soundfile as sf

# Download model
model_dir = snapshot_download(
    repo_id="azeddinShr/Spark-TTS-Arabic-Complete",
    local_dir="./arabic_model"
)

# Setup environment
sys.path.insert(0, './Spark-TTS/cli')
from SparkTTS import SparkTTS

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts = SparkTTS("./arabic_model", device)

# Generate speech
text = "مَرْحَبًا بِكُمْ فِي نَمُوذَجِ تَحْوِيلِ النَّصِّ إِلَى كَلَامٍ."
ref_text = "النَّصُّ الْمُطَابِقُ لِلصَّوْتِ الْمَرْجِعِيِّ"

wav = tts.inference(
    text,
    prompt_speech_path="reference.wav",
    prompt_text=ref_text
)

sf.write("output.wav", wav, samplerate=16000)
```

### Using Your Own Fine-tuned Model

Replace the model directory with your fine-tuned checkpoint:

```python
tts = SparkTTS("/path/to/your/finetuned_model", device)
```

---

## 📊 Training Details

### Dataset
- **Source:** [MBZUAI/ClArTTS](https://huggingface.co/datasets/MBZUAI/ClArTTS)
- **Size:** 12 hours (9,500 utterances)
- **Training subset:** 30% (~2,850 samples)
- **Language:** Classical Arabic (MSA)
- **Speaker:** Single male speaker

### Configuration
- **Base Model:** SparkAudio/Spark-TTS-0.5B (LLM only)
- **Method:** Full fine-tuning
- **Epochs:** 20
- **Batch Size:** 8 (effective)
- **Learning Rate:** 2e-4
- **Optimizer:** AdamW
- **Hardware:** Single NVIDIA GPU (Colab)

### Architecture
Only the **LLM (Qwen2)** component is fine-tuned. The BiCodec and wav2vec2 remain unchanged from the base model.

---

## ⚠️ Important Notes

### Input Requirements
- **Text must include Arabic diacritics (tashkeel)**
- Example: `مَرْحَبًا` not `مرحبا`
- Use AI tools or [online diacritizers](https://tahadz.com/mishkal)

### Reference Audio Requirements
- Duration: 5-30 seconds
- Quality: Clean, clear speech
- Language: Arabic (MSA/Classical preferred)

---

## 🔗 Resources

- **Pre-trained Model:** [HuggingFace](https://huggingface.co/azeddinShr/Spark-TTS-Arabic-Complete)
- **Base Model:** [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)
- **Dataset:** [ClArTTS](https://huggingface.co/datasets/MBZUAI/ClArTTS)
- **Original Spark-TTS:** [GitHub](https://github.com/SparkAudio/Spark-TTS)

---

## 📄 License

Apache 2.0 (same as base model)

---

## 🙏 Acknowledgments

- **SparkAudio Team** - Base Spark-TTS model
- **MBZUAI** - ClArTTS dataset
- **Hugging Face** - Transformers, Axolotl frameworks

---

## 📧 Contact

**Azeddin Sahir**
- Email: azdinsahir11@gmail.com
- HuggingFace: [@azeddinShr](https://huggingface.co/azeddinShr)

---

## 📝 Citation

```bibtex
@misc{spark-tts-arabic-2025,
  author = {Azeddin Sahir},
  title = {Spark-TTS Arabic: Fine-tuned on ClArTTS},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/spark-tts-arabic}
}
```