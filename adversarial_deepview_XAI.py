from NLP_helper_scripts.Multi_Task_Data_load import *
from NLP_helper_scripts.Multi_Task_model import *
from transformers import AutoTokenizer
from safetensors.torch import load_model
from datasets import load_dataset, concatenate_datasets
from evaluate import load
import pandas as pd


single_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
task = "sst2"


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
sentence1_key, sentence2_key = task_to_keys[task]
def multi_preprocess_function(examples):
    if sentence2_key is None:
       return  multi_tokenizer(examples[sentence1_key], max_length=128,padding='max_length',truncation=True)
    return multi_tokenizer(examples[sentence1_key], examples[sentence2_key], max_length=128,padding='max_length',truncation=True)


def single_preprocess_function(examples):
    if sentence2_key is None:
       return  single_tokenizer(examples[sentence1_key], max_length=128,padding='max_length',truncation=True)
    return single_tokenizer(examples[sentence1_key], examples[sentence2_key], max_length=128,padding='max_length',truncation=True)

model_args = ModelArguments(model_name_or_path="bert-base-uncased")
training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir="./models/mrpc_cola_rte/5epoch_bert",
    learning_rate=2e-5,
    num_train_epochs=5,
    overwrite_output_dir=True,
    remove_unused_columns=False,
)

multi_tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)


device = torch.device("cuda:0")

multi_tasks = [Task(id=0, name='cola', type='glue', num_labels=2),
 Task(id=1, name='mnli', type='glue', num_labels=3),
 Task(id=2, name='mrpc', type='glue', num_labels=2),
 Task(id=3, name='qnli', type='glue', num_labels=2),
 Task(id=4, name='qqp', type='glue', num_labels=2),
 Task(id=5, name='rte', type='glue', num_labels=2),
 Task(id=6, name='sst2', type='glue', num_labels=2)]

multi_model = MultiTaskModel(model_args.model_name_or_path, tasks=multi_tasks).to(device)
# model.load_state_dict(torch.load("./models/GLUE/all_tasks_and_data/bert/checkpoint-350000/model.safetensors"))
load_model(multi_model,"./models/multi-task-model/checkpoint-350000/model.safetensors")

actual_task = "mnli" if task == "mnli-mm"else task
dataset = load_dataset("glue", actual_task)
metric = load("glue", actual_task)

multi_preprocessed_dataset = dataset.map(multi_preprocess_function, batched=True)
single_preprocessed_dataset = dataset.map(single_preprocess_function, batched=True)

from NLP_helper_scripts.getData import prepare_label

validation_key = (
    "validation_mismatched"
    if task == "mnli-mm"
    else "validation_matched"
    if task == "mnli"
    else "validation"
)

merged = concatenate_datasets([multi_preprocessed_dataset["train"], multi_preprocessed_dataset[validation_key]])


def bert_glue_encode(dataset):
    # Convert batch of encoded features to numpy array.
    input_ids = np.array(dataset["input_ids"], dtype="int32")
    attention_masks = np.array(dataset["attention_mask"], dtype="int32")
    token_type_ids = np.array(dataset["token_type_ids"], dtype="int32")
    labels = np.array(dataset["label"], dtype="int32")

    #add check for test set since they may not have labels
    return (input_ids, attention_masks, token_type_ids) ,labels

# m_x_train, y_train = bert_glue_encode(multi_preprocessed_dataset["train"])
m_x_val, y_val = bert_glue_encode(merged)
# x_val, y_val = bert_glue_encode(single_preprocessed_dataset[validation_key])

import numpy as np

#Get the embedded data
n_samples = 10000
# sample_ids = np.random.choice(len(x_val[1]), n_samples)
sample_ids = np.array(list(range(58220,58220+n_samples)))

import gc

num_splits = 100
multi_sample = []
task = [6 for i in range(len(y_val))]
for i in range(num_splits):
    if i != (len(range(num_splits)) - 1):
        multi_input_ids = torch.LongTensor(
            m_x_val[0][sample_ids][(i * int(n_samples / num_splits)):(i + 1) * int(n_samples / num_splits)]).to(device)
        multi_attention_mask = torch.LongTensor(
            m_x_val[1][sample_ids][(i * int(n_samples / num_splits)):(i + 1) * int(n_samples / num_splits)]).to(device)
        multi_token_type_ids = torch.LongTensor(
            m_x_val[2][sample_ids][(i * int(n_samples / num_splits)):(i + 1) * int(n_samples / num_splits)]).to(device)
        # multi_labels = np.array(multi_val_data['labels'])
        multi_task_ids = torch.LongTensor(
            task[(i * int(n_samples / num_splits)):(i + 1) * int(n_samples / num_splits)]).to(device)
    else:
        multi_input_ids = torch.LongTensor(m_x_val[0][sample_ids][(i * int(n_samples / num_splits)):int(n_samples)]).to(
            device)
        multi_attention_mask = torch.LongTensor(
            m_x_val[1][sample_ids][(i * int(n_samples / num_splits)):int(n_samples)]).to(device)
        multi_token_type_ids = torch.LongTensor(
            m_x_val[2][sample_ids][(i * int(n_samples / num_splits)):int(n_samples)]).to(device)
        # multi_labels = np.array(multi_val_data['labels']2
        multi_task_ids = torch.LongTensor(task[(i * int(n_samples / num_splits)):int(n_samples)]).to(device)

    multi_sample_embedding = multi_model.embed(multi_input_ids, multi_attention_mask, multi_token_type_ids,
                                               task_ids=multi_task_ids)
    multi_sample_np = multi_sample_embedding.detach().cpu().numpy()
    multi_sample.append(multi_sample_np)

    del multi_sample_embedding
    gc.collect()
    torch.cuda.empty_cache()

multi_sample = np.array(multi_sample).reshape((n_samples, 768))
multi_Y1 = y_val[sample_ids]



from datasets import load_dataset, Dataset
sets = ['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte']

sst2_dataset = load_dataset('json', data_files='data/dev_ann.json', field=sets[0])
# sst2_dataset = load_dataset('json', data_files='dev_ann.json', field=sets[2])
task = "sst2"

num_classes = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
if task == "stsb":
    num_classes = 1
elif task.startswith("mnli"):
    num_classes = 3
else:
    num_classes = 2

def preprocess_og_sentence_function(examples):
    if examples['original_sentence'] != None:
        return single_tokenizer(examples["original_sentence"], max_length=128, padding='max_length', truncation=True)

def preprocess_perturbed_sentence_function(examples):
    return single_tokenizer(examples["sentence"], max_length=128, padding='max_length', truncation=True)

import pandas as pd
sst2_og_df = sst2_dataset["train"].to_pandas()

sst2_og_df = sst2_og_df.dropna()


cleaned_sst2 = datasets.Dataset.from_pandas(sst2_og_df)

preprocessed_og_sentence_dataset = cleaned_sst2.map(preprocess_og_sentence_function, batched=True)
preprocessed_pert_sentence_dataset = cleaned_sst2.map(preprocess_perturbed_sentence_function, batched=True)

perturb = "word"
og_sent_df = preprocessed_og_sentence_dataset.to_pandas()
og_word_perts = og_sent_df.where(og_sent_df["data_construction"] == "word").dropna()
og_sentence_perts = og_sent_df.where(og_sent_df["data_construction"] == "sentence").dropna().reset_index()

pert_sent_df = preprocessed_pert_sentence_dataset.to_pandas()
pert_word_perts = pert_sent_df.where(pert_sent_df["data_construction"] == "word").dropna()
sentence_perts = og_sent_df.where(og_sent_df["data_construction"] == "sentence").dropna().reset_index()
d = datasets.Dataset.from_pandas(sentence_perts)

task = [ 6 for i in range(len(preprocessed_pert_sentence_dataset))]

og_input_ids = torch.LongTensor(og_sentence_perts['input_ids']).to(device)
og_attention_mask = torch.LongTensor(og_sentence_perts['attention_mask']).to(device)
og_token_type_ids = torch.LongTensor(og_sentence_perts['token_type_ids']).to(device)
labels = np.array(og_sentence_perts['label'])

task_ids = torch.LongTensor(task).to(device)

pert_input_ids = torch.LongTensor(sentence_perts['input_ids']).to(device)
pert_attention_mask = torch.LongTensor(sentence_perts['attention_mask']).to(device)
pert_token_type_ids = torch.LongTensor(sentence_perts['token_type_ids']).to(device)

pert_sample_embedding = multi_model.embed(pert_input_ids, pert_attention_mask, pert_token_type_ids, task_ids=task_ids)

pert_X = pert_sample_embedding.detach().cpu().numpy()
pert_Y = labels

del pert_sample_embedding
gc.collect()
torch.cuda.empty_cache()

sent_attacks = np.array(sentence_perts["method"].map({'SCPN':0, 'T3':1,
                                       'StressTest':2, 'CheckList':3}))

pert_samples = 1
pert_sample_ids = np.random.randint(4 +17+19, 4 +17+19+ len(pert_X[sent_attacks == 3]), pert_samples)
total_X = np.append(multi_sample,pert_X[pert_sample_ids]).reshape(((len(multi_sample)+len(pert_X[pert_sample_ids])),768))
total_Y = np.append(multi_Y1, pert_Y[pert_sample_ids])


def pred_wrapper_multitask(x):
    softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        x = np.array(x, dtype=np.float32)
        tensor = torch.from_numpy(x).to(device)
        logits, loss = multi_model.output_heads[str(6)](tensor)
        probabilities = softmax(logits).detach().cpu().numpy()
    return probabilities

def data_viz1(sample, point, p, t, sample_id):
    if sample_id <= len(sample_ids):
        input = multi_preprocessed_dataset['validation']['sentence'][int(sample_ids[sample_id])]
    else:
        input = d['sentence'][pert_sample_ids[0]]
    return input


from deepview.DeepView import DeepView

# d1 = DeepViewk(pred_wrapper, classes, max_samples, batch_size, data_shape,
#        N, lam, resolution, cmap, interactive, title, use_case=use_case)


#--- Deep View Parameters ----
classes = np.arange(2)
batch_size = 64
max_samples = 10005
data_shape = (768,)
resolution = 100
N = 10
lam = 1
cmap = 'tab10'
metric = 'cosine'
disc_dist = (
    False
    if lam == 1
    else True
)

# to make sure deepview.show is blocking,
# disable interactive mode
interactive = False
title = 'movie-reviews BERT'

multi_og_deepview_task_cosine = DeepView(pred_wrapper_multitask, classes, max_samples, batch_size, data_shape,
                      N, lam, resolution, cmap, interactive, title, data_viz=data_viz1, metric=metric, disc_dist=disc_dist)

multi_og_deepview_task_cosine.add_samples(total_X,total_Y)


multi_og_deepview_task_cosine.show()



