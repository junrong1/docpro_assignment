import pandas as pd
from transformers import (BertTokenizer,
                          BertForSequenceClassification,
                          AdamW,
                          get_cosine_schedule_with_warmup,
                          EarlyStoppingCallback,
                          Trainer,
                          TrainingArguments)
from datasets import Dataset, load_metric
from sklearn.preprocessing import LabelEncoder
from app.topic_classifier.supervised.train.training_monitor import PrinterCallback
import torch


class FinetuneModel(object):

    def __init__(self, train_data_path="twitter_training.csv", test_data_path="twitter_validation.csv"):
        # We use an open source data twitter as an example
        self.train_df = pd.read_csv(train_data_path)
        self.test_df = pd.read_csv(test_data_path)

    @staticmethod
    def freeze_bert_layers(model, num_hidden_layers_to_unfreeze=2):
        # Freeze all layers
        for param in model.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last `num_hidden_layers_to_unfreeze` hidden layers
        for i in range(-1, -(num_hidden_layers_to_unfreeze + 1), -1):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Unfreeze the classification layer
        for param in model.classifier.parameters():
            param.requires_grad = True

    @staticmethod
    def compute_metrics(eval_pred):
        accuracy_metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = torch.tensor(logits).argmax(dim=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return accuracy

    @staticmethod
    def get_optimizer_params(model, training_args):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def preprocess(self):

        def tokenize_function(examples):
            return tokenizer(examples['content'], padding='max_length', truncation=True, max_length=512)

        self.train_df.columns = ["idx", "type", "label", "content"]
        self.test_df.columns = ["idx", "type", "label", "content"]

        self.train_df.reset_index(drop=True, inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)

        # Filter useless data
        train_df = self.train_df.loc[self.train_df["label"] != "Irrelevant"]
        test_df = self.test_df.loc[self.test_df["label"] != "Irrelevant"]

        label_encoder = LabelEncoder()
        train_df['label'] = label_encoder.fit_transform(train_df['label'])
        test_df['label'] = label_encoder.fit_transform(test_df['label'])

        train_df.dropna(inplace=True, subset=["label", "content"])
        test_df.dropna(inplace=True, subset=["label", "content"])

        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Format datasets for PyTorch
        train_dataset = train_dataset.remove_columns(["content", "__index_level_0__"])
        test_dataset = test_dataset.remove_columns(["content", "__index_level_0__"])
        train_dataset.set_format("torch")
        test_dataset.set_format("torch")

        return train_dataset, test_dataset

    def __call__(self):
        train_dataset, test_dataset = self.preprocess()
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=self.train_df['label'].nunique())
        FinetuneModel.freeze_bert_layers(model)
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./bert-finetune",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=64,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir="./logs",
            learning_rate=4e-5,
            load_best_model_at_end=True
        )

        # Create optimizer and scheduler
        optimizer_grouped_parameters = FinetuneModel.get_optimizer_params(model, training_args)
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * training_args.num_train_epochs * len(
                train_dataset) // training_args.per_device_train_batch_size,
            num_training_steps=training_args.num_train_epochs * len(
                train_dataset) // training_args.per_device_train_batch_size
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=FinetuneModel.compute_metrics,
            optimizers=(optimizer, scheduler),
            callbacks=[PrinterCallback(), EarlyStoppingCallback(early_stopping_patience=2,
                                                                early_stopping_threshold=0
                                                                )],
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")

