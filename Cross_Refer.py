from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import os

def load_model(model_name):
    """
    Load the pre-trained model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def read_cv_file(cv_file):
    """
    Read the CV file content and parse it into sections
    """
    with open(cv_file, 'r') as file:
        content = file.read()
    
    education_list = []
    experience_list = []
    skill_list = []
    sector_list = []
    
    overall_list = [education_list, experience_list, skill_list, sector_list]
    
    content = content.split('\n')
    for i in content:
        if i == "## Education:":
            curr = 0
        elif i == "## Experience:":
            curr = 1
        elif i == "## Skills:":
            curr = 2
        elif i == "## Sector:":
            curr = 3
        elif i == "## Timeline:":
            break
        else:
            i = i.replace("- ", "").replace(":: ", "").strip('\n ')
            if i != "":
                overall_list[curr].append(i)
    
    return overall_list

def read_lor_files(lor_folder):
    """
    Read all LOR files in the folder
    """
    lor_list = []
    for filename in os.listdir(lor_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(lor_folder, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                lor_list.append(content)
            print(f"Added LOR File: {filename}")
    return lor_list

def calculate_trust_score(experience_list, lor_list, tokenizer, model):
    """
    Calculate the trust score for each experience in the CV
    """
    temp_trust = []
    for i_ in experience_list:
        inputs_1 = tokenizer(i_, return_tensors='pt', truncation=True, padding=True)
        temp_temp = []
        for k in lor_list:
            inputs_2 = tokenizer(k, return_tensors='pt', truncation=True, padding=True)
            
            with torch.no_grad():
                embeddings_1 = model(**inputs_1).last_hidden_state.mean(dim=1)
                embeddings_2 = model(**inputs_2).last_hidden_state.mean(dim=1)
                
                # Calculate Cosine Similarity
                cos_sim = cosine_similarity(embeddings_1, embeddings_2)
                temp_temp.append(cos_sim)
        
        temp_trust.append(max(temp_temp))
    
    return temp_trust

def calculate_min_avg_trust(temp_trust):
    """
    Calculate the minimum and average trust scores
    """
    min_trust = float(min(temp_trust).item()) if temp_trust else 0.0
    avg_trust = float(sum(temp_trust).item() / len(temp_trust)) if temp_trust else 0.6
    return min_trust, avg_trust

def crossref(path, folder):
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    tokenizer, model = load_model(model_name)
    
    CV_file = path  # Single CV file
    LOR_folder = folder  # Folder containing recommendation letters
    
    overall_list = read_cv_file(CV_file)
    lor_list = read_lor_files(LOR_folder)
    
    temp_trust = calculate_trust_score(overall_list[1], lor_list, tokenizer, model)
    min_trust, avg_trust = calculate_min_avg_trust(temp_trust)
    return avg_trust
