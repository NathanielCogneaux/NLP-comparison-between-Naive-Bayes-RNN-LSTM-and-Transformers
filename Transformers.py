# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset["train"])

# Define stopwords
en_stopwords = set(stopwords.words("english"))

# Define preprocessing function
def preprocessing(sentence):
    sentence = sentence.lower() # Remove caps
    sentence = re.sub(r"[^a-z\s]", "", sentence) # Remove everything that is not a letter or a space
    sentence = word_tokenize(sentence) # Tokenize
    sentence = [word for word in sentence if word not in en_stopwords] # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    sentence = [lemmatizer.lemmatize(word) for word in sentence] # Lemmatize
    return " ".join(sentence)

# Preprocess the sentences
df["sentence"] = df["sentence"].apply(preprocessing)

# Print some preprocessed samples
print(df["sentence"].head())

# Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define tokenization function
def tokenize(batch):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=512)

# Tokenize the dataset
dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset["train"]))
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Split the dataset into training, validation and testing datasets
length = len(dataset["train"])
train_val_dataset = dataset["train"].select(range(int(0.8*length)))
test_dataset = dataset["train"].select(range(int(0.8*length), length))

# Further split the train_val_dataset into training and validation datasets
train_indices, val_indices = train_test_split(
    list(range(len(train_val_dataset))), test_size=0.2
)

train_dataset = train_val_dataset.select(train_indices)
val_dataset = train_val_dataset.select(val_indices)

# Print data splits
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Use the validation dataset for evaluation during training
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the training and validation sets
training_accuracy = trainer.evaluate(train_dataset)
validation_accuracy = trainer.evaluate(val_dataset)

print(f"Training accuracy: {training_accuracy['eval_accuracy']}")
print(f"Validation accuracy: {validation_accuracy['eval_accuracy']}")

# Evaluate the model on the test set
evaluation_results = trainer.evaluate(test_dataset)

# Print the evaluation results to check what keys are present
print(evaluation_results)

# Print the accuracy
print(f"Accuracy on the test set: {evaluation_results['eval_accuracy']}")
