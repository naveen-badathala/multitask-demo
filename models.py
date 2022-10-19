import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer

def inference(input_txt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'roberta-large' #uncased should have do_lower_case=True
    # model = AutoModel.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True) 
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
#     num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    #input_text = "The kind of anger rages like a sea in storm"
    input_text = input_txt
    inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
    model = AutoModelForSequenceClassification.from_pretrained("hypo-meta-ST", output_attentions=True) 
    #model.cuda()
    outputs = model(inputs.to(device))  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
    pred_label = torch.sigmoid(outputs[0])
    return pred_label



def inference_st_hyperbole(input_txt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'roberta-large' #uncased should have do_lower_case=True
    # model = AutoModel.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True) 
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
#     num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    #input_text = "The kind of anger rages like a sea in storm"
    input_text = input_txt
    inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
    model = AutoModelForSequenceClassification.from_pretrained("hypo-hyperbole-ST", output_attentions=True) 
    #model.cuda()
    outputs = model(inputs.to(device))  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
    pred_label = torch.sigmoid(outputs[0])
    return pred_label



def inference_st_metaphor(input_txt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'roberta-large' #uncased should have do_lower_case=True
    # model = AutoModel.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True) 
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
#     num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    #input_text = "The kind of anger rages like a sea in storm"
    input_text = input_txt
    inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
    model = AutoModelForSequenceClassification.from_pretrained("hypo-meta-ST", output_attentions=True) 
    #model.cuda()
    outputs = model(inputs.to(device))  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
    pred_label = torch.sigmoid(outputs[0])
    return pred_label
