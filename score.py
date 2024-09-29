import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def read_input(file_path):
    experiences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("## Experience:"):
                current_section = 'experience'
            elif current_section == 'experience' and line.startswith('-'):
                experiences.append(line[1:].strip())
            if line.startswith("## Skills:"):
                break
    
    return experiences

# New function to read skills from skills_output.txt
def read_skills(file_path):
    skills = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            skills.append(line.strip())  # Add each line as a skill to the list
    return skills

# Function to split experience into heading and description based on "::"
def split_experience(experiences):
    split_experiences = []
    for experience in experiences:
        if '::' in experience:
            parts = experience.split('::')
            heading = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ''
            split_experiences.append((heading, description))
        else:
            split_experiences.append((experience, ''))  # No description present
    
    return split_experiences

# Function to remove stopwords from text
def remove_stopwords(text):
    tokens = re.findall(r'\w+', text.lower())  # Find all words
    filtered_tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return ' '.join(filtered_tokens)

# Function to process experiences and skills by removing stopwords
def preprocess_data(experiences, skills):
    processed_experiences = [(remove_stopwords(exp[0]), remove_stopwords(exp[1])) for exp in experiences]
    processed_skills = [remove_stopwords(skill) for skill in skills]
    return processed_experiences, processed_skills

# Function to compute relevance scores for each skill against each experience
def compute_relevance_scores(experience_file_path, skill_file_path):
    # Load the pre-trained model
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    # Read experiences and skills
    experiences = read_input(experience_file_path)
    skills = read_skills(skill_file_path)
    
    # Split experiences into heading and description
    split_experiences = split_experience(experiences)

    # Remove stopwords from experiences and skills
    processed_experiences, processed_skills = preprocess_data(split_experiences, skills)

    # Create embeddings for skills and combined experience (heading + description)
    experience_texts = [' '.join(exp) for exp in processed_experiences]
    experience_embeddings = model.encode(experience_texts, convert_to_tensor=True)
    skill_embeddings = model.encode(processed_skills, convert_to_tensor=True)

    # Calculate cosine similarity between each skill and each experience
    relevance_matrix = np.zeros((len(skills), len(experiences)))

    for i, skill_embedding in enumerate(skill_embeddings):
        similarities = util.cos_sim(skill_embedding, experience_embeddings)
        relevance_matrix[i] = similarities.cpu().numpy()

    return relevance_matrix, processed_skills, experience_texts

def print_relevance_matrix(relevance_matrix, skills, experiences):
    print(f'{"Skill":<30} | ' + ' | '.join([f"Exp {i+1}" for i in range(len(experiences))]) + ' | RMS')
    print('-' * (35 + 15 * len(experiences) + 10))  # Adjust the width for RMS column
    
    rms_above_threshold = 0  
    
    for i, skill in enumerate(skills):
        scores = ' | '.join([f"{score:.4f}" for score in relevance_matrix[i]])
        rms = np.sqrt(np.mean(relevance_matrix[i]**2))  # Calculate RMS for the row
        
        if rms > 0.47:  
            rms_above_threshold += 1
        
        print(f'{skill:<30} | {scores} | {rms:.4f}')  # Print RMS for each skill
    
    print(f'\nNumber of skills with RMS > 0.47: {rms_above_threshold}')

def get_skills_with_high_rms_and_combined_l2(relevance_matrix, skills, threshold=0.43):
    high_rms_skills = []
    combined_relevance_scores = []  
    for i, skill in enumerate(skills):
        rms = np.sqrt(np.mean(relevance_matrix[i]**2))  
        
        if rms > threshold:  
            high_rms_skills.append(skill)
            combined_relevance_scores.append(relevance_matrix[i])  # Collect the relevance scores
    
    # Stack relevance scores from all high RMS skills
    combined_relevance_scores = np.concatenate(combined_relevance_scores, axis=0)
    
    # Calculate the L2 norm for all combined relevance scores
    combined_l2_norm = np.linalg.norm(combined_relevance_scores)
    
    return high_rms_skills, combined_l2_norm



def score(experience_file_path, skill_file_path, rms_threshold=0.47):
    
    # Compute the relevance matrix
    relevance_matrix, processed_skills, experience_texts = compute_relevance_scores(experience_file_path, skill_file_path)
    
    # Get skills with RMS above the threshold and calculate the combined L2 norm
    high_rms_skills, combined_l2_norm = get_skills_with_high_rms_and_combined_l2(relevance_matrix, processed_skills, threshold=rms_threshold)
    
    return high_rms_skills, combined_l2_norm
