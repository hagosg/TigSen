# TigSen
Tigrigna sentiment Analysis Dataset
This repository contains Python 3 code for performing Sentiment Analysis on Tigrigna social media texts using various deep learning and transformer-based models. The goal of this project is to classify the sentiments of posts or comments written in the Tigrigna language, one of the widely spoken languages in Eritrea and Northern Ethiopia.

📂 Repository Structure
bash
Copy
Edit
Tigrigna-Social-Media-Texts-Sentiment-Analysis/ <br>
│
├── README.md                        # Project overview and usage instructions <br>
├── TigSent_CNN/                     # CNN-based sentiment classification  <br>
├── TigSent_LSTM/                    # LSTM-based sentiment classification <br>
├── TigSent_XLM_R/                   # XLM-Roberta-based model (Transformer) <br>
├── TigSent_mBERT/                   # Multilingual BERT-based model <br>
├── TigSent_LLaMA/                   # Multilingual Cross transfer model <br>
├── TigSen.xlsx               # Labeled dataset of Tigrigna texts <br>
🔍 Project Objectives <br>
- Fine-tune state-of-the-art NLP models to classify Tigrigna text sentiment as:

  - Positive 😃

  - Negative 😠

  - Neutral 😐

- Explore and compare the performance of:

  
  - Convolutional Neural Networks (CNN)

  
  - Long Short-Term Memory networks (LSTM)

  
  - XLM-RoBERTa transformer

  
  - Multilingual BERT (mBERT)
  - Multilingual LLaMA

📊 Dataset
The dataset used is:
- Tigrigna_Social_Media_DataSet.xlsx: A manually labeled dataset of Tigrigna sentences collected from social media platforms.
-  Each entry contains:

  - Tigrigna sentence

  - Sentiment label (Positive, Negative, Neutral)


🧠 Models Overview
🧱 TigSent_CNN
A convolutional neural network model built using Keras/TensorFlow for sentence classification.

🔁 TigSent_LSTM
An LSTM-based architecture to capture sequential dependencies in Tigrigna sentences.

🌍 TigSent_XLM_R
Fine-tuned XLM-Roberta model, which supports many languages including Tigrigna.

🧠 TigSent_mBERT
Uses Multilingual BERT, pre-trained on 100+ languages including Tigrigna, fine-tuned for sentiment classification.
⚙️ Installation and Setup
Clone the repository

bash
git clone https://github.com/hagosg/TigSen.git
cd Tigrigna-Sentiment-Analysis
Create a virtual environment (optional)

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies


pip install -r requirements.txt  # You may need to create this if not present
Install HuggingFace Transformers (for XLM-R/mBERT models)

pip install transformers datasets
🚀 Running the Models
Each model is in its own folder. You can run the scripts independently.

cd TigSent_CNN
python train_cnn.py
Replace with appropriate script names for:

  - train_lstm.py

  - train_xlm_r.py

  - train_mbert.py

Make sure to adjust paths for dataset loading in each script.

📈 Evaluation
Each model reports:

   - Accuracy

  - Precision

  - Recall

 - F1-Score

 - Confusion matrix

Optional: You can implement model comparison across architectures.

🌐 Language Support
All models are fine-tuned to support the Tigrigna language. Tokenization and preprocessing steps are customized to account for Tigrigna script and linguistic nuances.

🤝 Contributing
Feel free to:

   - Submit pull requests

   - Open issues

   - Suggest improvements (especially on tokenization or model enhancements for low-resource Tigrigna NLP)

📜 License
This project is under an open research license. If you use the models or data in your work, please cite the repository and acknowledge the contributors.

📧 Contact
For questions or collaborations:

   - Maintainer: hagosg

   - Email: (hagosg81@bit.edu.cn)

