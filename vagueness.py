import os
import fitz
from groq import Groq

printable = set(list(' ,0123456789-=~!@#$%^&*()_+abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,:;"\'/\n[]\\{}|?><.'))
bloat = {'per', 'and', 'but', 'the', 'for', 'are', 'was', 'were', 'be', 'been', 'with', 'you', 'this', 'but', 'his',
         'from', 'they', 'say', 'her', 'she', 'will', 'one', 'all', 'would', 'there', 'their', 'what', 'out', 'about',
         'who', 'get', 'which', 'when', 'make', 'can', 'like', 'time', 'just', 'into', 'year', 'your', 'good', 'some',
         'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
         'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
         'any', 'these', 'give', 'day', 'most'}

def cleaned(t):
    return "".join([i for i in t if i in printable])

def pdf_to_text(pdf):
    with fitz.open(pdf) as doc:
        ina = [cleaned(page.get_text()).split("\n") for page in doc]
        texta = []
        for a in ina:
            texta += a
    return [k for k in texta if len(k) > 2 and not k.isspace() and k not in bloat]

def vague(folder_path, output_folder):
    client = Groq(api_key="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")

    folder_path = 'D:/app/REC'  # Adjust path to your folder
    output_folder = 'D:/app/lor_outputs'
    
    
    # Loop through each PDF file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):  # Make sure it's a PDF file
            input_pdf = os.path.join(folder_path, file_name)
            output_file = os.path.join(output_folder, f'{file_name[:-4]}_vague.txt')

            # Convert the PDF to text
            context = pdf_to_text(input_pdf)
            context = ' '.join(context)

            # Define the prompt
            prompt = prompt = """This text is a Letter Of Recommendation towards someone and needs to be checked for vague or exaggerated statements within the text. Only consider very short phrases of 2-4 words while scanning, use surrounding context extensively. It is not necessary that the LOR's contain such text, be specific when identifying vagueness and exaggeration. Do not consider basic praise statements as exaggerations, only flag excessively hyperbolic text as exaggerated in this context. Do NOT point out for the sake of pointing something out, be clear on exactly why it is vague or exaggerated. Only consider very short phrases of 2-4 words while scanning, use surrounding context extensively. For vagueness, remember that basic praises do not have to be incredibly precise with supporting arguments, however if larger claims are made without any evidence, flag it. Beware that context or evidence may be found on lines adjacent to the phrase you might flag, look through all data carefully. DO NOT FLAG DATA AS VAGUE JUST FOR THE SAKE OF FLAGGING, I want very specific claims with no evidence to be flagged as vague. Only consider very short phrases of 2-4 words while scanning, use surrounding context extensively, it is not necessary for this document to have vague or exaggerated claims, your answer may be blank, in that case output ALLOK in one line and nothing else. USE SURROUNDING CONTEXT ALWAYS, IF THERE IS ANY EVIDENCE, EVEN ANECDOTAL SUPPORTING A POTENTIALLY VAGUE CLAIM, CONSIDER IT TO BE NOT VAGUE
Examples of Vagueness:
1) great potential : Vague
2) incredible enthusiasm : Exaggerated
For each phrase which seems vague or exaggerated, display it on a new line in the format:
TYPE :: PHRASE :: SEVERITY :: REASON
TYPE = VAG / EXA depending on which type of phrase this piece of text is
PHRASE = THE EXACT WORDING INCLUDING DELIMITERS AND WHITESPACES OF THE PHRASE WHICH HAS BEEN FLAGGED
SEVERITY = HIGH / LOW depending on how vague or exaggerated you think the phrase is, IT CANNOT BE ANY OTHER VALUE LIKE MEDIUM OR VERY HIGH OR VERY LOW, IT CAN ONLY BE ONE OF HIGH OR LOW
REASON = SINGLE PHRASE REASON as to why you flagged it, be VERY EXACT and EXPLICIT with your WORDING, IT MUST BE A SHORT PHRASE, MAKE SURE NOT TO INCLUDE PHRASES JUST BECAUSE OF A PLAUSIBLE REASON, THEY MUST ACTUALLY BE VAGUE OR EXAGGERATED
STICK STRICTLY TO THE FORMAT, THIS DATA WILL BE PROCESSED BY PYTHON AND MUST FOLLOW THE FORMAT EXACTLY, DO NOT ADD EXPOSITION OR EXPLANATIONS, SIMPLY THE OUTPUT AND A SINGLE LINE FOR EVERY PHRASE
"""

            # Generate the completion using Groq's API
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

            # Collect the output from the completion
            text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    text += chunk.choices[0].delta.content

            # Write the output to a file
            with open(output_file, 'w') as f:
                f.write(text)

            print(f"Finished processing {file_name}, output written to {output_file}")


