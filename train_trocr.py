! pip install -q evaluate jiwer
#df = import and call the modeule creating df 
train_df =df.iloc[:-100].reset_index(drop=True)
test_df = df.iloc[-100:].reset_index(drop=True)#.head(4)


import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

from transformers import TrOCRProcessor
from transformers import AutoFeatureExtractor, AutoTokenizer

source_model= input('enter the source model path e.g. mohammadalihumayun/trocr-ur')
processor = TrOCRProcessor.from_pretrained(source_model)

train_dataset = IAMDataset(root_dir=rt,
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir=rt,
                           df=test_df,
                           processor=processor)



from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained(source_model)
#model = VisionEncoderDecoderModel.from_pretrained('cxfajar197/urdu-ocr')
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


from evaluate import load
cer_metric = load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

from huggingface_hub import HfFolder #, HfApi
hg_token=input('enter hugging face token')
HfFolder.save_token(hg_token)

from transformers import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

repo_name = input('enter target repo path')
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
    output_dir="./"+repo_name,
    logging_steps=1000,
    save_steps=3000,
    eval_steps=1000,
    num_train_epochs=2,
    report_to='none',
    push_to_hub=True,    
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()
#trainer.evaluate()

kwargs = {
    "language": "ur",
    "model_name": "trocr for Urdu",  # a 'pretty' name for your model
    "finetuned_from": "cxfajar197/urdu-ocr",
    "tasks": "ocr",
}
trainer.push_to_hub(**kwargs)
processor.push_to_hub(repo_id="mohammadalihumayun/"+repo_name)
