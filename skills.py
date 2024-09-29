# from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import pipeline
import os
# import google.generativeai as genai
import fitz
from groq import Groq

printable = set(list(' ,0123456789-=~!@#$%^&*()_+abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,:;"\'/\n[]\\{}|?><.'))
bloat = {'per', 'and', 'but', 'the', 'for', 'are', 'was', 'were', 'be', 'been', 'with', 'you', 'this', 'but', 'his',
         'from', 'they', 'say', 'her', 'she', 'will', 'one', 'all', 'would', 'there', 'their', 'what', 'out', 'about',
         'who', 'get', 'which', 'when', 'make', 'can', 'like', 'time', 'just', 'into', 'year', 'your', 'good', 'some',
         'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
         'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
         'any', 'these', 'give', 'day', 'most'}
# load_dotenv()
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# genai.configure(api_key=GEMINI_API_KEY)
oracle = pipeline("question-answering", model="deepset/roberta-base-squad2")
theta = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


def delimiter(phraset):
    l0 = 1
    for i in range(3):
        if phraset[i][1] > 0.62:
            l0 += 1
    l1 = l0 + 1
    for i in range(10):
        if phraset[l0 + i][1] > 0.60:
            l1 += 1
    l2 = l1 + 1
    for i in range(20):
        if phraset[l1 + i][1] >= 0.52:
            l2 += 1

    i1 = [x[0] for x in phraset[:l0]]
    i2 = [x[0] for x in phraset[l0:l1]]
    i3 = [x[0] for x in phraset[l1:l2]]
    return [i1, i2, i3]


def phrases_by_relevance(text, prompts):
    qembed = theta.encode([*prompts])
    embeddings = theta.encode(text)
    result = [tensor.item() for tensor in list(cos_sim(qembed, embeddings)[0])]
    phraset = [(text[i], result[i]) for i in range(len(text))]
    phraset.sort(key=lambda x: x[1], reverse=True)
    return phraset


def cleaned(t):
    return "".join([i for i in t if i in printable])


def pdf_to_text(pdf):
    with fitz.open(pdf) as doc:
        ina = [cleaned(page.get_text()).split("\n") for page in doc]
        texta = []
        for a in ina:
            texta += a
    return [k for k in texta if len(k) > 2 and not k.isspace() and k not in bloat]


# model = genai.GenerativeModel("gemini-1.5-flash")


def process_pdf_skills(path,out):
    client = Groq(api_key="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")

    # Directly read from 'output.txt'
    with open(path, "r") as file:
        file2 = file.read()

    # Extract the skills section from the file
    file2 = file2.split("## Skills")[1]
    file2 = file2.split("## Sector")[0]
    file2 = file2.strip()
    context = file2

    prompt = prompt = """\n\nPROMPT:\nAbove is a list of skills relevant in a professional context, they are found on a processed resume. From this sort and return only the skills which are relevant and can be classified as skills. remove all other skills, return as a list. Some relevant skills may be a set of words without any whitespaces, add whitespaces accordingly,
Example: microsoftword will become microsoft word
ONLY USE SKILLS PROVIDED ABOVE, DO NOT CREATE YOUR OWN, USE EXAMPLES BELOW ONLY FOR REFERENCE, NEW DATA MUST NOT BE GENERATED, FOLLOW THE FORMAT STRICTLY
DO NOT PROVIDE EXPLANATIONS OR EXPOSITION IN YOUR OUTPUT, DO NOT ADD REASONS FOR INCLUDING OR NOT INCLUDING SKILLS. THERE SHOULD BE NO ELEMENTS IN THE ARRAY WHICH ARE NOT PRESENT IN THE ORIGINAL SKILL LIST
YOUR LIST SHOULD NOT CONTAIN REASONS TO JUSTIFY WHY SOMETHING DOES NOT MATCH WITH EXAMPLES, SUCH AS
- Microsoft is not present so do not include it
INSTEAD SIMPLY OMIT THAT ENTRY AND DONT PROVIDE REASONING. NO MATTER WHAT, THERE SHOULD BE NO REASONING AND ONLY ELEMENTS BELONGING TO THE SKILL LIST ORIGINALLY PRESENT IN YOUR OUTPUT
Your output must contain only the final list of valid skills, it must be in square brackets, each separated with commas. All text will be in lowercase. 
Examples of skills to remove:
- client
- clients
- focus
- determination
- hiring
- instructor
- summary
- quality
- trainer
- process
- enterprise
Examples of skills to keep:
- performance management
- adobe photoshop
- microsoft word
- therapist
- leadership
- photography
- latin
- author
- accounting
- consulting
- sales
DO NOT INCLUDE TEXT LIKE "microsoft is not present so i will include presentations", SIMPLY INCLUDE "presentations" IN THE LIST
"""

    completion = client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=[
            {
                "role": "user",
                "content": context + "\n\n" + prompt
            }
        ],
        temperature=0.04,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    # Collect the output text from the response
    text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            text += chunk.choices[0].delta.content

    # Write the output to a new file
    with open(out, "w") as f:
        f.write(text)
    
    print("Finished processing .")
    
    
# process_pdf_skills("D:/app/input.txt", "D:/app/output2.txt")




