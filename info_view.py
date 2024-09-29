    


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
            elif line.startswith("## Skills:"):
                break
            
    
    return experiences

def extract_sector_from_file(filename):
    # Read the contents of the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Iterate through the lines and look for the "## Sector:" keyword
    for i in range(len(lines)):
        if lines[i].strip() == "## Sector:":
            # The actual sector value is on the line below
            sector = lines[i + 1].strip()
            return sector[2:]
        
def extract_experience_headings(file_path):
    # Initialize an empty list to store the headings
    headings = []
    experiences = read_input(file_path)
    # Loop through each experience in the list
    for experience in experiences:
        # Split the experience at the "::" and take the first part as the heading
        heading = experience.split("::")[0].strip()
        headings.append(heading)
    
    return headings



