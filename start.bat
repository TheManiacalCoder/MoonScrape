@echo off
echo [MODEL INITIALIZATION]
echo Checking existing model structure...

:: Only download if missing
python -c "from pathlib import Path; from transformers import BertModel, BertTokenizer; model_path = Path('fine_tuned_bert_model'); print('\nChecking model files...'); (BertModel.from_pretrained('bert-base-uncased').save_pretrained(model_path) if not model_path.exists() else print('\nUsing existing model')); BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained(model_path); print('\nModel ready')"

echo [STARTING MAIN APPLICATION]
python SERP_Scraper.py

pause