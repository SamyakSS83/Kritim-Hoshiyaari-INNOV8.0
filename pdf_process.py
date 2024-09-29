# from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import pipeline
import os
import google.generativeai as genai
import fitz

printable = set(list(' ,0123456789-=~!@#$%^&*()_+abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,:;"\'/\n[]\\{}|?><.'))
bloat = {'per', 'and', 'but', 'the', 'for', 'are', 'was', 'were', 'be', 'been', 'with', 'you', 'this', 'but', 'his',
         'from', 'they', 'say', 'her', 'she', 'will', 'one', 'all', 'would', 'there', 'their', 'what', 'out', 'about',
         'who', 'get', 'which', 'when', 'make', 'can', 'like', 'time', 'just', 'into', 'year', 'your', 'good', 'some',
         'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
         'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
         'any', 'these', 'give', 'day', 'most'}
# load_dotenv()
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key='AIzaSyDxqNrW1jO0dlErlEj0B5pqUfpjZ4h1wjY')
oracle = pipeline("question-answering", model="deepset/roberta-base-squad2")
theta = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


def delimiter(phraset):
    l0 = 1
    for i in range(3):
        if phraset[i][1]>0.62:
            l0 += 1
    l1 = l0 + 1
    for i in range(10):
        if phraset[l0+i][1]>0.60:
            l1 += 1
    l2 = l1 + 1
    for i in range(20):
        if phraset[l1+i][1] >= 0.52:
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
    return [k for k in texta if len(k)>2 and not k.isspace() and k not in bloat]


model = genai.GenerativeModel("gemini-1.5-flash")


def process_pdf(path):
    prompt_timeline = """\n\n
This is a resume of a person. Your output will be passed into python and MUST follow a fixed FORMAT, there should be no exposition, or extra text in your output. It must ALWAYS confine to the format. Do not deviate from the specified format.
FORMAT:
## Education:
- Name of degree1 :: College/University1
- Name of degree2 :: College/University2

## Experience:
- Job Title1 :: DESCRIPTION1
- Job Title2 :: DESCRIPTION2

## Skills:
- Skill1
- Skill2

## Sector:
- Sector

## Timeline:
- TYPE1 :: START-TIME1 -- END-TIME1 :: EVENT1 :: POSITION1
- TYPE2 :: START-TIME2 -- END-TIME2 :: EVENT2 :: POSITION2

END OF FORMAT
DO NOT INCLUDE ANY OTHER INFORMATION IN YOUR OUTPUT, DO NOT INCLUDE ANY OTHER DELIMITERS OR INDEXES, DO NOT USE COMMA SEPARATED VALUES, DO NOT INCLUDE ANY OTHER TEXT IN YOUR OUTPUT

In the format of a list of skills, remove any whitespaces from the start and end of skills, do not change their text and make sure that EVERY SINGLE ONE OF THEM IS FINDABLE AS TEXT IN THE RESUME
For Experience category, IGNORE THE NAMES OF THE ORGANIZATIONS IF PRESENT
For Sector category, the sector in which the person has experience in, either obtained from the first line of the CV, or from the context of the CV if the first line is vague

If any of the points are not available, such as name of company/organization don't mention it.
This response will be processed with python, stick strictly to the format, use '##' before every heading as mentioned in the format

Format of START-TIME: MM-YYYY (IF MONTH IS UNSPECIFIED, MM = 00, Example, if year is 2009 and month is not specified, START-TIME: 00-2009)
Format of END-TIME: MM-YYYY (IF MONTH IS UNSPECIFIED, MM = 00, Example, if year is 2009 and month is not specified, END-TIME: 00-2009)
ALWAYS ALWAYS ALWAYS INCLUDE BOTH START AND END TIMES, EVEN IF THEY ARE CURRENT, DO NOT MISS ANY DATES, REMEMBER ALL FORMATS
IF THE DATE IS UNKNOWN, USE 00-0000, DO NOT OMIT START OR END TIMES, ALWAYS INCLUDE BOTH
DO NOT MISS ANY DATES, REMEMBER ALL FORMATS
Format of EVENT: (A string containing the name of the event)
Format of TYPE: 
i) If the event is a job, TYPE=JOB, if the job is for less than 3 months, TYPE=INT
ii) If the event is an event, TYPE=EVE, example: "Attended the 2012 World Economic Forum"
iii) If the event is being a member of something, TYPE=MEM, example: "Board Member of Advocacy 4 Kids"
iv) If the event is education OF THE PERSON THAT WROTE THE RESUME, TYPE=EDU, this should be from a univerisiy or college, not awards or recognitions
v) If the event is an award or recognition, TYPE=AWD. Example: Most Valuable Player or Example: Employee of the Month
vi) For anything else, TYPE=UKN
DO NOT MISS ANY EVENT TYPES, REMEMBER ALL 7 categories (JOB, INT, EVE, MEM, EDU, AWD, UKN). USE UNKNOWN IF IT IS PRESENT
If any of these times are "current", format will be CURRENT
FOR DATES WHICH ARE NOT SPECIFIED, USE 00-0000
For POSITION, the possible values ARE ONLY HIGH, MEDIUM AND LOW for JOB TYPE EVENTS
Examples of HIGH: CEO, President, Director, Manager, Head of Department, CFO, Dean, CTO
Examples of MEDIUM: Supervisor, Team Leader, Senior Associate, Senior Analyst, Senior Consultant, Senior Engineer
Examples of LOW: Associate, Analyst, Consultant, Engineer, Assistant, Intern, Trainee, Volunteer, Junior Associate, Junior Analyst, Junior Consultant, Junior Engineer
FOR ALL OTHER TYPES OF EVENTS, POSITION SHOULD BE LOW ALWAYS, ONLY FOR JOB TYPE EVENTS, POSITION CAN BE HIGH, MEDIUM OR LOW
THERE SHOULD BE NO OTHER ELEMENTS IN EACH LINE, LINES SHOULD NOT BE INDEXED OR CONTAIN ANY OTHER DELIMITERS OTHER THAN THE ONES SPECIFIED, FOLLOW THE FORMAT STRICTLY
Example: - EDU :: 01-2011 :: B.E in Computer Science from ABCD University
Example: - MEM :: 01-2010 -- 00-2017 :: Board Member of Advocacy 4 Kids
Example: - JOB :: 00-2009 -- CURRENT :: Organizer and Capacity Building Strategist
DO NOT ASSUME DATA ON YOUR OWN, DO NOT CHANGE ANY DATA IN THE RESUME. INCLUDE ALL TIMES AND DATES IN YOUR RESPONSE. DO NOT EXCLUDE IMPORTANT DATA IN THE RESUME
"""
    input = pdf_to_text("D:/aries/INNOV8-2.0-Finals-main/INNOV8-2024-Finals-main/INNOV8-2024-Finals-main/Final_Resumes/Resume_of_ID_0.pdf")
    context = input
    context = ' '.join(context)
    # response = model.generate_content(context + prompt_parser, generation_config=genai.types.GenerationConfig(temperature=0.15))
    response2 = model.generate_content(context + prompt_timeline, generation_config=genai.types.GenerationConfig(temperature=0.1))
    # print(response.text)
    with open(path, 'w') as f:
        f.write(response2.text)
    print(f"Output done")

