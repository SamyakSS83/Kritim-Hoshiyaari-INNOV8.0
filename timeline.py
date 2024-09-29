import re
import os

timeline = []

def parse_timeline(file_name):
    global timeline
    with open(file_name, 'r') as file:
        lines = file.readlines()
        in_timeline = False
        for line in lines:
            if line.strip() == '## Timeline:':
                in_timeline = True
            elif in_timeline and line.strip() == '':
                break
            elif in_timeline:
                match = re.match(r'-\s*(\w+)\s*::\s*(\d{2}-\d{4}\s*--\s*\d{2}-\d{4}|\d{2}-\d{4}\s*--\s*CURRENT|\d{2}-\d{4})\s*::\s*(.*)\s*::\s*(\w+)', line)
                if match:
                    date_range = match.group(2).split('--')
                    entry = {
                        'type': match.group(1).strip(),
                        'start_date': date_range[0].strip(),
                        'end_date': date_range[1].strip() if len(date_range) > 1 else 'CURRENT',
                        'text': match.group(3).strip(),
                        'severity': match.group(4).strip()
                    }
                    timeline.append(entry)

def calculate_factor():
    max_date = '00-0000'
    job_map = {}
    education_map = {}
    jobeducation_map = {}
    latest_education_year = '0000'
    
    for entry in timeline:
        start_date, end_date = entry['start_date'], entry['end_date']
        if start_date == 'CURRENT':
            start_date = '12-2017'
        if end_date == 'CURRENT':
            end_date = '12-2017'
        
        start_month, start_year = start_date.split('-')
        end_month, end_year = end_date.split('-')
        
        if entry['type'] == "JOB":
            for year in range(int(start_year), int(end_year) + 1):
                job_map[year] = job_map.get(year, 0) + 1
                jobeducation_map[year] = jobeducation_map.get(year, 0) + 1
                
        elif entry['type'] == "EDU":
            for year in range(int(start_year), int(end_year) + 1):
                education_map[year] = education_map.get(year, 0) + 1
                jobeducation_map[year] = jobeducation_map.get(year, 0) + 1
            latest_education_year = max(latest_education_year, end_year)
    
    sum_jobs = sum(job_map.values())
    sum_education = sum(education_map.values())
    sum_jobeducation = sum(jobeducation_map.values())

    factor1 = sum_jobs / len(job_map) if job_map else 1
    factor2 = sum_education / len(education_map) if education_map else 1
    factor3 = sum_jobeducation / len(jobeducation_map) if jobeducation_map else 1

    factor = factor1 * factor2 * factor3

    flag, bigflag = 0, 0
    for entry in timeline:
        if entry['type'] == "JOB" and entry['severity'] == 'HIGH' and int(entry['start_date'].split('-')[1]) in {int(latest_education_year), int(latest_education_year) + 1}:
            flag = 1

        if entry['type'] == "EDU":
            for entry2 in timeline:
                if entry2['type'] == "JOB" and entry2['severity'] == 'HIGH' and int(entry2['start_date'].split('-')[1]) == int(entry['end_date'].split('-')[1]):
                    bigflag = 1

    vacancies = 0
    first_year = min(job_map.keys(), default=0)
    last_year = max(job_map.keys(), default=0)
    for year in range(first_year, last_year + 1):
        if year not in jobeducation_map:
            vacancies += 1
    vacancy_factor = 1 + (vacancies / len(job_map) if job_map else 1)

    return factor, vacancy_factor, flag, bigflag

def timeli(path):
    global timeline
    timeline = []

    file_name = path  # Single file for output
    parse_timeline(file_name)

    factor, vacancy_factor, flag, bigflag = calculate_factor()

    # with open(file_name, 'w') as output_file:
    #     output_file.write(f'{factor:.4f} {vacancy_factor:.4f} {flag} {bigflag}\n')

    # Print corresponding three values
    return factor, vacancy_factor, flag
    
    
    


if __name__ == '__main__':
    path = 'D:/app/input.txt'
    print(timeli(path))
    
