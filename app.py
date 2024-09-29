import dash
from dash import dcc, html, Input, Output
import pandas as pd
import os
import torch
import base64  # For decoding uploaded files
from pdf_process import *
from skills import *
from score import *
from Cross_Refer import *
from info_view import *
from timeline import *
from trust import *
from vagueness import *

app = dash.Dash(__name__, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"])

# Helper function to save uploaded files
def save_uploaded_file(contents, filename):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open(filename, 'wb') as f:
            f.write(decoded)
        return filename
    except Exception as e:
        print(f"Error saving file {filename}: {e}")
        return None

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1("Resume and LOR Analysis Dashboard", style={'textAlign': 'center', 'color': '#007BFF'}),
        dcc.Upload(
            id='resume-upload',
            children=html.Button('Upload Resume (PDF)', className='btn btn-primary'),
            multiple=False
        ),
        dcc.Upload(
            id='lor-upload',
            children=html.Button('Upload Letters of Recommendation (ZIP)', className='btn btn-primary'),
            multiple=False
        ),
        html.Div(id='output-data-upload', style={'marginTop': '20px'})
    ], style={'textAlign': 'center'}),

    # Wrap results section in dcc.Loading for loading animation
    dcc.Loading(
        id="loading",
        type="default",  # Change to "circle" or "dot" for different styles
        children=[
            html.Div(id='results-section', style={'marginTop': '20px'})
        ]
    )
], style={'backgroundColor': '#F8F9FA', 'padding': '20px'})

@app.callback(
    Output('output-data-upload', 'children'),
    Output('results-section', 'children'),
    Input('resume-upload', 'contents'),
    Input('lor-upload', 'contents')
)
def update_output(resume_contents, lor_contents):
    if resume_contents is None or lor_contents is None:
        return "Please upload your resume and letters of recommendation.", ""

    # Save uploaded files
    resume_path = save_uploaded_file(resume_contents, 'resume.pdf')
    lor_path = save_uploaded_file(lor_contents, 'lor.zip')

    if not resume_path or not lor_path:
        return "Error saving files. Please try again.", ""

    try:
        # Process PDFs and gather results
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        process_pdf(resume_path)
        process_pdf_skills(resume_path, "output1.txt")
        skill, s = score(resume_path, "output1.txt")
        f1, f2, f3 = timeli(resume_path)
        sec = extract_sector_from_file(resume_path)
        head = extract_experience_headings(resume_path)
        trust = crossref(resume_path, "D:/app/REC")
        a, b = trust_check(resume_path, "D:/app/REC", "D:/app/lor_outputs")

        # Ensure skills are a list
        skill_list = skill.split(", ") if isinstance(skill, str) else skill  # Convert string to list if necessary

        # Create results section
        results = html.Div([
            html.H3("Candidate Details", className='text-info'),
            html.Div([
                html.P(f"Sector: {sec}"),
                html.H4("Experience Headings:", className='text-warning'),
                html.Ul([html.Li(h) for h in head]),
                html.H4("Skills:", className='text-warning'),
                html.Ul([html.Li(s) for s in skill_list]),  # Ensure skills are in a list
                html.H4("Quality Score:", className='text-success'),
                html.P(f"{s:.2f}"),
                html.H3("Verification", className='text-info'),
                html.P(f"LOR Trust Score: {trust:.2f}"),
                html.P(f"Resume Timeline Consistency Score: {(2 - f2) * 100:.2f}%"),
                html.P(f"Chance of Fraud: {a:.2f}"),
            ], style={'padding': '10px', 'border': '1px solid #007BFF', 'borderRadius': '5px', 'backgroundColor': '#E9ECEF'})
        ], style={'marginTop': '20px'})
        
        return "Files uploaded successfully!", results

    except Exception as e:
        return f"Error during processing: {e}", ""

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.run_server(debug=True)
