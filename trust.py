import pandas as pd
import math
import os
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
from timeline import *

# Load the tokenizer and model globally (to avoid reloading in each function)
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def calculate_vagueness(folder_vague):
    """Calculate the vagueness factor for all LOR files in a folder."""
    vague_values = []
    for filename in os.listdir(folder_vague):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_vague, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_read:
                content = file_read.read()

            vague_value = 0
            total_len = 0

            for l, i in enumerate(content.split('\n')):
                total_len = l
                k = i.split(" ")
                if "LOW" in k:
                    vague_value += 0.5
                else:
                    vague_value += 1

            vague_value = vague_value / (total_len + 1)
            vague_values.append(vague_value)

    # Calculate the overall vagueness factor
    if vague_values:
        overall_vague_value = sum(vague_values) / len(vague_values)
    else:
        overall_vague_value = 0
    return overall_vague_value

def parse_cv(path):
    """Parse the CV file into sections."""
    with open(path, 'r') as file:
        content = file.read()

    # Initialize lists to store parsed sections
    education_list = []
    experience_list = []
    skill_list = []
    sector_list = []

    overall_list = [education_list, experience_list, skill_list, sector_list]

    # Parse the CV content into respective sections
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

def calculate_trust(cv_experience, lor_folder):
    """Calculate trust value based on cosine similarity between CV experience and LORs."""
    temp_trust = []
    lor_list = []

    # Load all LOR files
    for filename in os.listdir(lor_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(lor_folder, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lor_list.append(file.read())

    # Compare CV experience with LORs using cosine similarity
    for exp in cv_experience:
        inputs_1 = tokenizer(exp, return_tensors='pt', truncation=True, padding=True)
        temp_temp = []
        for lor in lor_list:
            inputs_2 = tokenizer(lor, return_tensors='pt', truncation=True, padding=True)

            with torch.no_grad():
                embeddings_1 = model(**inputs_1).last_hidden_state.mean(dim=1)
                embeddings_2 = model(**inputs_2).last_hidden_state.mean(dim=1)

            cos_sim = cosine_similarity(embeddings_1, embeddings_2)
            temp_temp.append(cos_sim.item())

        if temp_temp:
            temp_trust.append(max(temp_temp))

    # Calculate trust values
    if temp_trust:
        csv_trust = [float(min(temp_trust)), float(sum(temp_trust) / len(temp_trust))]
    else:
        csv_trust = [0.0, 0.6]  # Default trust values in case no similarity was found

    return csv_trust[1]  # Return average trust value

def trust_check(path, folder_lor, folder_vague):
    """Main function to calculate and return the Untrustworthy Factor."""
    # Load the timeline data
    risk_factor, vacancy_factor, flag = timeli(path)

    # Calculate vagueness factor
    overall_vague_value = calculate_vagueness(folder_vague)


    # Parse the CV file
    overall_list = parse_cv(path)

    # Calculate trust value
    trust_value = calculate_trust(overall_list[1], folder_lor)  # Use experience section
    print("Trust Value:", trust_value)

    # Calculate the Untrustworthy factor
    Untrustworthy_factor = (
        overall_vague_value * 0.1
        + 0.2 * math.tanh(risk_factor / 5)
        + 0.1 * math.tanh(vacancy_factor - 1)
        + 0.3 * flag
    )
    Untrustworthy_factor += 0.4 * (1 - trust_value)

    return Untrustworthy_factor, overall_vague_value

# Example usage:
# untrustworthy_factor = trust("D:/app/input.txt", "D:/app/REC", "D:/app/lor_outputs")
# print("Final Untrustworthy Factor:", untrustworthy_factor)
