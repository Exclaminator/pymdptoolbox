import matplotlib.pyplot as plt
import csv
import numpy as np

# DEBUG instructions: Unexpected results? Check the following things:
# Is the row filter correct?
# Is the folder set?

# settings
folder = "20190611-170535"

#only select rows with the given value for the specified columns (usefull when you have changed multiple values
row_filter = {
    'beta':'0.2',
    'delta':'0.2'
}

xColumn = 'var'
yColumns = ['average_value', 'min_value', 'variance']

labelColumn = 'mdp_id'

# init arrays
results = {}
x = []

with open('..\\..\\logs\\' + folder + '\\results_params.log', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # check if row should be read
        use_row = True
        for k in row_filter.keys():
            if row[k] != row_filter[k]:
                use_row = False
                break
        if not use_row:
            continue

        if row['problem_type'] not in results:
            results[row['problem_type']] = {}
        problemResults = results[row['problem_type']]
        mdp = row[labelColumn]
        if mdp not in problemResults:
            problemResults[mdp] = []

        problemResults[mdp].append(row)

        if len(x) == 0 or x[-1] != float(row[xColumn]):
            x.append(float(row[xColumn]))



for key in yColumns:
    for problem in results:
        for mdp in results[problem]:
            plot_values = [float(d[key]) for d in results[problem][mdp]]

            plt.plot(x, plot_values, label=mdp)


    plt.legend()
    plt.title(problem + " " + key)
    plt.xlabel(xColumn)
    plt.ylabel(key)
    #plt.axis([0, 1, 0, 30])
    plt.show()
