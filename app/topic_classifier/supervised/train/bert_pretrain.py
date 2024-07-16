from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from app.topic_classifier.supervised.train.training_monitor import PrinterCallback

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class PretrainModel(object):

    def __init__(self, train_data_path="twitter_training.csv", test_data_path="twitter_validation.csv"):
        # We use an open source data twitter as an example
        self.train_df = pd.read_csv(train_data_path)
        self.test_df = pd.read_csv(test_data_path)

    def preprocess(self):

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        self.train_df.columns = ["idx", "type", "label", "content"]
        self.test_df.columns = ["idx", "type", "label", "content"]

        self.train_df.reset_index(drop=True, inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)

        train_df = self.train_df.loc[self.train_df["label"] != "Irrelevant"]
        test_df = self.test_df.loc[self.test_df["label"] != "Irrelevant"]

        train_df.dropna(inplace=True, subset=["label", "content"])
        test_df.dropna(inplace=True, subset=["label", "content"])

        train_content_df = pd.DataFrame({"text": train_df["content"].tolist()})
        test_content_df = pd.DataFrame({"text": test_df["content"].tolist()})

        train_dataset = Dataset.from_pandas(train_content_df)
        test_dataset = Dataset.from_pandas(test_content_df)

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        return train_dataset, test_dataset

    def __call__(self):
        train_dataset, test_dataset = self.preprocess()
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        training_args = TrainingArguments(
            output_dir="./bert-pretrain",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=8,
            save_steps=10_000,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=200,
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[PrinterCallback()]
        )

        trainer.train()

