import networkx as nx
import csv
import matplotlib.pyplot as plt

# Create an empty directed graph
G = nx.DiGraph()

# Read the CSV file and add edges to the graph
with open('Final_Persons_And_Recommenders.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        user_id = int(row['ID'])
        recommenders_ids = list(map(int, row['Recommenders ID'].strip('[]').split(',')))
        for recommender_id in recommenders_ids:
            G.add_edge(user_id, recommender_id)

# Function to generate cycles
def generate_cycles(G, node, path, cycles):
    if len(path) > 5:
        return
    for neighbor in G.neighbors(node):
        if neighbor in path:
            if neighbor == path[0]:
                cycles.add(tuple(path + [neighbor]))
        else:
            generate_cycles(G, neighbor, path + [neighbor], cycles)

# Find all cycles of size <= 4
cycles = set()
for node in G.nodes():
    generate_cycles(G, node, [node], cycles)

#set of number of people in atleast one cycle
people = set()

#sort cycles by length
cycles = sorted(cycles, key=len)
for cycle in cycles:
    for person in cycle:
        people.add(person)
print(len(people))

peopleincycleofsize2 = set()
peopleincycleofsize3 = set()
peopleincycleofsize4 = set()
peopleincycleofsize5 = set()
allpeople = set()

#dict for (number of cycles a person is in)^2/(total length of cycles he is in)
# Initialize the dictionary to store counts and lengths
person_cycle_stats = {}
# Iterate through each cycle and update the dictionary
for cycle in cycles:
    cycle_length = len(cycle)
    for person in cycle:
        if person not in person_cycle_stats:
            person_cycle_stats[person] = {'count': 0, 'length': 0}
        person_cycle_stats[person]['count'] += 1
        person_cycle_stats[person]['length'] += cycle_length

final_person_cycle_stats = {}
for person in person_cycle_stats:
    final_person_cycle_stats[person] = (person_cycle_stats[person]['count'] ** 2) / person_cycle_stats[person]['length']


# Print the dictionary
print(person_cycle_stats)

for cycle in cycles:
    if len(cycle) == 2:
        for person in cycle:
            if person not in allpeople:
                peopleincycleofsize2.add(person)
                allpeople.add(person)
    if len(cycle) == 3:
        for person in cycle:
            if person not in allpeople:
                peopleincycleofsize3.add(person)
                allpeople.add(person)
    if len(cycle) == 4:
        for person in cycle:
            if person not in allpeople:
                peopleincycleofsize4.add(person)
                allpeople.add(person)
    if len(cycle) == 5:
        for person in cycle:
            if person not in allpeople:
                peopleincycleofsize5.add(person)
                allpeople.add(person)

print(len(allpeople))

print(len(peopleincycleofsize3))
print(" ".join(map(str, peopleincycleofsize3)))

print(len(peopleincycleofsize4))
print(" ".join(map(str, peopleincycleofsize4)))

print(len(peopleincycleofsize5))
print(" ".join(map(str, peopleincycleofsize5)))

# Print the cycles
# for cycle in cycles:
#     print(cycle)

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()