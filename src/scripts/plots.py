import matplotlib.pyplot as plt
import csv
import numpy as np

value = []
robust = []
ep = []
ml = []
x = []

with open('C:\\Users\\jens\\Documents\\studie\\pymdptoolbox\\logs\\20190606-231826\\results_var.log', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        if (row[' beta'] == ' 0.4'):
            if row[' mdp_id'] == " valueIteration":
                value.append(row)
                x.append(float(row[' delta']))
            if row[' mdp_id'] == " robust interval":
                robust.append(row)
            if row[' mdp_id'] == " robust max_like":
                ml.append(row)
            if row[' mdp_id'] == " robust ellipsoid":
                ep.append(row)

key = " average_value"
ep_avg = [float(d[key]) for d in ep]
ml_avg = [float(d[key]) for d in ml]
v_avg  = [float(d[key]) for d in value]

plt.plot(x, ep_avg, 'r-', label='interval')
plt.plot(x, ml_avg, 'b-', label='value iteration')
plt.legend()
plt.title("Min")
plt.xlabel("Max fluctuation in P")
plt.ylabel("Value")
plt.axis([0, 1, 0, 30])

plt.show()
test = 45