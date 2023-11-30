!pip install transformers
import json  # You forgot to import json in your original code

def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa[access]:  # Changed this to 'access' to match your logic
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers

# Paths to your existing SQuAD files in Colab. Just make sure these paths are correct.
train_file_path = 'answers.json'  # Or whatever the path to the train file is
val_file_path = 'test.json'  # Or whatever the path to the validation file is
train_contexts, train_questions, train_answers = read_squad('answers.json')
val_contexts, val_questions, val_answers = read_squad('test.json')
def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    # this means the answer is off by 'n' tokens
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
                    
add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
tokenizer.decode(train_encodings['input_ids'][0])
train_encodings.keys()
def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift one token forward
        go_back = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end']-go_back)
            go_back +=1
    # update our encodings object with the new token-based start/end positions
    encodings.update({
        'start_positions': start_positions,
        'end_positions': end_positions
        })

# apply function to our data
add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

# PyTorch Fine-tuning
import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)
from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
!pip install optuna
import optuna
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch

def objective(trial):
    # Setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Hyperparameters to be optimized
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    epochs = trial.suggest_int('epochs', 1, 10)

    # Your model should be defined before this
    model.to(device)
    model.train()

    # Initialize Adam optimizer with weight decay
    optim = AdamW(model.parameters(), lr=lr)

    # Initialize data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Add scheduler for learning rate
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs
    )

    for epoch in range(epochs):
        loop = tqdm(train_loader)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    return loss.item()  # Replace with your preferred metric

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')  # Use 'maximize' for metrics like accuracy
    study.optimize(objective, n_trials=100)
!pip install optuna
import optuna
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch

def objective(trial):
    # Setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Hyperparameters to be optimized
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    epochs = trial.suggest_int('epochs', 1, 10)

    # Your model should be defined before this
    model.to(device)
    model.train()

    # Initialize Adam optimizer with weight decay
    optim = AdamW(model.parameters(), lr=lr)

    # Initialize data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Add scheduler for learning rate
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs
    )

    for epoch in range(epochs):
        loop = tqdm(train_loader)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    return loss.item()  # Replace with your preferred metric

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')  # Use 'maximize' for metrics like accuracy
    study.optimize(objective, n_trials=100)
#this is the previous code before auto hypermeters tuning

# -*- coding: utf-8 -*-
"""
**Recommendation**: Run training on a GPU.
If you are using Colab: Enable this in the menu "Runtime" > "Change Runtime type" > Select "GPU" in dropdown.
Then change the `use_gpu` arguments below to `True`
"""

# Make sure you have a GPU running
!nvidia-smi

# Install the latest release of Haystack in your own environment
! pip install farm-haystack

# Install the latest master of Haystack
!pip install --upgrade pip
!pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab]

from haystack.nodes import FARMReader

!pip install sentence_transformers
!pip install farm-haystack[inference]
from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
data_dir = "/content/"
# data_dir = "PATH/TO_YOUR/TRAIN_DATA"

reader.train(data_dir=data_dir, train_filename="answers.json", use_gpu=True, batch_size=32, n_epochs=5,learning_rate= 1e-5, save_dir="my_model",)

# Saving the model happens automatically at the end of training into the `save_dir` you specified
# However, you could also save a reader manually again via:
reader.save(directory="my_model")

# If you want to load it at a later point, just do:
new_reader = FARMReader(model_name_or_path="my_model")

reader_eval_results = new_reader.eval_on_file("/content", "answers.json", device="cuda")

reader_eval_results

context = '''<input text>'''

new_reader.predict_on_texts("what is the patient medical history?",[context])

"""###Inference Using Pipeline

"""

from haystack import Pipeline, Document
from haystack.utils import print_answers
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
p = Pipeline()
p.add_node(component=new_reader, name="Reader", inputs=["Query"])
res = p.run(
    query="<input query here> ", documents=[Document(content=context)]
)
print_answers(res,details="medium")
from haystack import Pipeline, Document
from haystack.nodes import FARMReader
from haystack.utils import print_answers
import logging
from haystack import Document
from haystack.nodes import FARMReader

# Suppress specific warnings from Haystack
logging.getLogger('haystack').setLevel(logging.ERROR)

# ... rest of your code

# Function to chunk large text into smaller parts
def chunk_text(text, max_words=400):
    sentences = text.split('. ')
    current_chunk = ""
    chunks = []

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_words:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '. '

    if current_chunk:
        chunks.append(current_chunk)  # Add the last chunk if any
    return chunks


# Function to process all chunks for a single question and retrieve the top 5 answers
def get_top_answers_above_threshold(question, chunks, reader, score_threshold=0.59):
    qualifying_answers = []
    for chunk in chunks:
        result = reader.predict(query=question, documents=[Document(content=chunk)], top_k=5)
        answers = result['answers']
        # Filter the answers based on the score threshold
        qualifying_answers.extend([answer for answer in answers if answer.score > score_threshold])
    return qualifying_answers
questions_and_top_answers = []

# Your large text goes here
large_text = """<INPUT TEXT> """

# Replace this with your actual list of 30 questions
questions = [
    "What is the patient’s diabetes type?",
"What is the diabetes duration?",
"What type of sleep disordered breathing was the patient diagnosed with?",
"What equipment does the patient use at home for sleep disordered breathing?",
"How many hospital admissions for heart failure did the patient have within the past 12 months?",
"Which test(s) served as the basis for new diabetes diagnosis?",
"What characterization of HF was present at admission or whenever first recognized?",
"What symptoms did the patient have at the time of admission?",
"What were the physical findings at the time of admission?",
"What are the ICD-10-CM other diagnosis codes?",
"What are the ICD-10-PCS principal procedure codes?",
"What are the ICD-10-PCS other procedure codes?",
"What was the highest level of mitral valve regurgitation found on an echocardiogram prior to discharge?",
"What type(s) of DVT prophylaxis were initiated by the end of hospital day 2?",
"On which did the patient receive the COVID-19 vaccination?",
"What was the patient’s cause of death?",
"When is the earliest documentation of comfort measures only?",
"What status best describes the patient’s heart failure symptoms present at the time of discharge compared to those closest to admission?",
"What were the physical findings closest to discharge or death?",
"What is the dosage of the ACEI medication prescribed at discharge?",
"What was the frequency in which the patient was asked to take the ACEI medication at discharge?",
"What was the dosage of the ARB medication prescribed at discharge?",
"What was the frequency in which the ARB medication was prescribed at discharge?",
"What was the dosage of the ARNI medication prescribed at discharge?",
"What was the frequency in which the ARNI medication was prescribed at discharge?",
"What was the dosage of the prescribed beta blocker medication?",
"What was the frequency of the prescribed beta blocker medication?",
"What was the dose of the prescribed SGLT2 inhibitor medication?",
"What is the frequency in which the prescribed SGLT2 inhibitor medication should be taken?",
"What was the dosage of MRA medication prescribed at discharge?",
"Was potassium ordered or scheduled after discharge from current hospitalization?",
"Was a renal function test scheduled after discharge?",
"What was the dose at which the anticoagulation therapy was prescribed?",
"What was the frequency at which the prescribed anticoagulant should be taken?",
"What was the specific anti-hyperglycemic medication prescribed at discharge?",
"What was the ASA dose prescribed at discharge?",
"What was the prescribed ASA frequency at discharge?",
"What was the dosage at which the other antiplatelet medication was prescribed at discharge?",
"What was the frequency at which the antiplatelet medication should be taken as prescribed at discharge?",
"What was the dosage of clopidogrel prescribed at discharge?",
"What was the frequency at which the patient was prescribed to take clopidogrel at discharge?",
"What was the dosage of the lipid lowering medication prescribed at discharge?",
"What was the prescribed frequency for the lipid lowering medication?",
"Which smoking cessation therapies were prescribed?",
"If no follow up visit was scheduled, is there is a medical or patient reason documented for no follow-up appointment being scheduled?",
"Was a follow-up phone call scheduled?",
"Was a follow-up appointment scheduled at discharge for diabetes management?",
"Did the patient receive counseling on the Therapeutic Lifestyle Changes (TLC) diet?",
"Did the patient receive obesity weight management treatment?",
"Did the patient receive a written activity recommendation or referral to cardiac rehabilitation?",
"Was the patient referred to an outpatient cardiac rehab program?",
"Did the patient receive Anticoagulation Therapy education?",
"Did the patient received diabetes teaching?",
"Did the patient receive instructions regarding PT/INR planned follow-up?",
"Was the patient referred to a sleep study for suspected sleep apnea?",
"Did the patient receive a referral to an outpatient HF management program?",
"What types of outpatient HF management programs was the patient referred to at discharge?",
"Was the patient provided information on how to access the AHA My HF Guide/Heart Failure Interactive Workbook?",
"Was an advanced care plan/surrogate decision maker documented or discussed during this hospitalization?",
"Was an advance directive executed?",
"Was a care transition record is transmitted to the next level of care provider no later than the seventh post-discharge day?",
"Is there documentation in the care transition record that includes the discharge medications, dosage, and indication for use or that no medications were prescribed at discharge?",
"Is there documentation in the care transition record that includes follow-up treatment(s) and service(s) needed?",
"Does the care transition record include procedures performed during the hospitalization?",
"Does the care transition record include the reason for hospitalization?",
"Is there documentation in the care transition record that includes treatment(s) and service(s) provided during hospitalization?",
"During this admission, was a standardized health related social needs form or assessment completed?",
"If yes, what are the areas of unmet social need?"

]

# Chunk the large text
chunks = chunk_text(large_text)

for question in questions:
    top_answers = get_top_answers_above_threshold(question, chunks, reader)
    question_and_answers = {"question": question, "answers": top_answers}
    questions_and_top_answers.append(question_and_answers)

# Print the summary of all questions with their qualifying answers
for qa in questions_and_top_answers:
    print(f"Question: {qa['question']}")
    if qa['answers']:
        for answer in qa['answers']:
            print(f"Answer: {answer.answer}\nScore: {answer.score}\n")
    else:
        print("No qualifying answers found.\n")

#for BoolQ type questions 
!pip install transformers
!pip install datasets
!pip install transformers[torch]
!pip install accelerate>=0.20.1
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
!pip install wandb
import wandb

# Initialize wandb
wandb.init(project="distilbert-boolq", name="basic-run")

def preprocess_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                record = json.loads(line)
                processed_record = {
                    "question": record["question"],
                    "passage": record["passage"],
                    "label": int(record["label"])  # Ensure labels are integers
                }
                data.append(processed_record)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line: {e}")
    return data

def convert_to_dataset(data, tokenizer):
    tokenized_inputs = tokenizer(
        [d["question"] + " " + d["passage"] for d in data],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = [d["label"] for d in data]
    dataset = Dataset.from_dict({**tokenized_inputs, "labels": labels})
    return dataset

# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# File paths should be correctly pointing to your files
train_file_path = '/content/boolq_questions_train.jsonl'
val_file_path = '/content/val_boolq_questions_.jsonl'

# Preprocess the data
train_data = preprocess_json(train_file_path)
val_data = preprocess_json(val_file_path)

# Convert to datasets
train_dataset = convert_to_dataset(train_data, tokenizer)
val_dataset = convert_to_dataset(val_data, tokenizer)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=32,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=10,
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False,
    report_to="wandb",  # Enable logging to wandb
    run_name="distilbert-boolq-experiment"  # Name of the wandb run
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model and the tokenizer
model.save_pretrained('models/distilbert_boolq')
tokenizer.save_pretrained('models/distilbert_boolq')

# Close wandb run
wandb.finish()
from transformers import pipeline

# Load the trained model and tokenizer
model_path = '/content/models/distilbert_boolq'  # Adjust to where you saved your model
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Create a pipeline for sequence classification
qa_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# The question and passage
question = "Does the patient have a known history of vaping or e-cigarette use in the past 12 months?"
passage = "<input text>"

# Tokenize and predict
result = qa_pipeline(question + " " + passage)

# Interpret the result
label_map = {0: 'False', 1: 'True'}
answer = label_map[int(result[0]['label'].split('_')[-1])]

print(f"Question: {question}")
print(f"Passage: {passage}")
print(f"Answer: {answer}")
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast

# Load the trained model and tokenizer
model_path = '/content/models/distilbert_boolq'  # Adjust to where you saved your model
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Create a pipeline for sequence classification with truncation and padding
qa_pipeline = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    truncation=True,
    padding=True
)

# Define a function to split the passage into chunks
def split_into_chunks(passage, max_chunk_length):
    # Tokenize the passage and cut it into chunks of max_chunk_length
    tokens = tokenizer.tokenize(passage)
    chunks = [' '.join(tokens[i:i + max_chunk_length]) for i in range(0, len(tokens), max_chunk_length)]
    return chunks
# The passage and list of questions
passage = "<input text>"
questions = [
    "Does the patient have a known history of vaping or e-cigarette use in the past 12 months?",
"Does the patient have a heart failure etiology of ischemic heart failure/CAD?",
"Does the patient have a history of non-ischemic heart failure etiology?",
"Did the patient have a medical history of heart failure prior to this admission?",
"Is the patient listed as a candidate for a heart transplant procedure?",
"Does the patient have atrial fibrillation at presentation and/or did they have it at any time during this hospitalization?",
"Is there documentation of new onset of atrial fibrillation from this admission?",
"Does the patient have atrial flutter at presentation and/or did they have it at any time during this hospitalization?",
"Is there documentation of new onset of atrial flutter from this admission?",
"Did the patient receive a new diagnosis of diabetes during this hospitalization?",
"Were there any other conditions contributing to heart failure exacerbation during this admission?",
"Did the patient have an active bacterial or viral infection at admission or during this hospitalization?",
"Was LVSD documented during this hospitalization?",
"Did the patient receive an intravenous vasoactive medication from the list provided during this hospitalization?",
"Was the patient ambulating by the end of hospital day 2?",
"Was DVT or PE documented during this hospitalization?",
"Did the patient have the influenza vaccination?",
"Did the patient have the COVID-19 vaccination?",
"Was the patient in a COVID-19 vaccine trial?",
"Did the patient receive a pneumococcal vaccine?",
"Was there a reason for not switching to ARNI at discharge?",
"Was potassium ordered or scheduled after discharge from current hospitalization?",
"Was a renal function test scheduled after discharge?",
"Are there any contraindications or documented reasons for not providing ivabradine at discharge?",
"Did the patient receive ICD counseling at discharge?",
"Has ICD therapy been placed prior to or during hospitalization, or is it documented in the medical record that ICD placement is planned post hospital discharge?",
"Was there a documented reason for not placing or prescribing ICD therapy at discharge?",
"Was CRT-D placed or prescribed at discharge?",
"Was CRT-P placed or prescribed at discharge?",
"Was there a documented reason for not placing or prescribing CRT therapy at time of discharge?",
"Was a follow-up phone call scheduled?",
"Was a follow-up appointment scheduled at discharge for diabetes management?",
"Did the patient receive counseling on the Therapeutic Lifestyle Changes (TLC) diet?",
"Did the patient receive obesity weight management treatment?",
"Did the patient receive a written activity recommendation or referral to cardiac rehabilitation?",
"Was the patient referred to an outpatient cardiac rehab program?",
"Did the patient receive Anticoagulation Therapy education?",
"Did the patient received diabetes teaching?",
"Did the patient receive instructions regarding PT/INR planned follow-up?",
"Was the patient referred to a sleep study for suspected sleep apnea?",
"Did the patient receive a referral to an outpatient HF management program?",
"Was the patient provided information on how to access the AHA My HF Guide/Heart Failure Interactive Workbook?",
"Was an advanced care plan/surrogate decision maker documented or discussed during this hospitalization?",
"Was an advance directive executed?",
"Was a care transition record is transmitted to the next level of care provider no later than the seventh post-discharge day?",
"Was a care transition included?",
"Is there documentation in the care transition record that includes the discharge medications, dosage, and indication for use or that no medications were prescribed at discharge?",
"Is there documentation in the care transition record that includes follow-up treatment(s) and service(s) needed?",
"Does the care transition record include procedures performed during the hospitalization?",
"Does the care transition record include the reason for hospitalization?",
"Is there documentation in the care transition record that includes treatment(s) and service(s) provided during hospitalization?"
    
]  # List of questions
# Calculate max_chunk_length considering the model's max length and the special tokens
max_model_length = tokenizer.model_max_length  # Usually 512 for DistilBert
max_chunk_length = max_model_length - 3  # for special tokens [CLS], [SEP], [SEP]

# Process each question
for question in questions:
    # Calculate the actual max chunk length by subtracting the question length
    question_tokens_length = len(tokenizer.tokenize(question))
    actual_max_chunk_length = max_chunk_length - question_tokens_length

    chunks = split_into_chunks(passage, actual_max_chunk_length)
    confident_answers = []

    for chunk in chunks:
        # Tokenize and predict
        result = qa_pipeline(question, chunk)

        # Filter results with confidence score above 0.60
        if result[0]['score'] > 0.2:
            label_map = {0: 'False', 1: 'True'}
            answer = label_map[int(result[0]['label'].split('_')[-1])]
            confident_answers.append((chunk, answer, result[0]['score']))

    # Print all confident answers for the current question
    if confident_answers:
        print(f"Question: {question}")
        for chunk, answer, score in confident_answers:
            print(f"Passage Chunk: {chunk[:50]}...")  # Show the first 50 characters of the chunk
            print(f"Answer: {answer}, Confidence: {score}")
    else:
        print(f"Question: {question}")
        print("No confident answer found.")