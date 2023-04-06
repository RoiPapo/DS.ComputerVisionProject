import matplotlib.pyplot as plt
import numpy as np

extraction_times = {'B0': 111,
                    'Baseline(B2)': 198,
                    'B6': 2526,
                    'B7': 4364}

def graph_1():

    accuracy = {'B0': 0.807, 'Baseline(B2)': 0.856, 'B6': 0.91, 'B7': 0.892}
    f1_10 = {'B0': 0.8113, 'Baseline(B2)': 0.8685, 'B6': 0.9202, 'B7': 0.9005}
    f1_25 = {'B0': 0.8015, 'Baseline(B2)': 0.86624, 'B6': 0.8926, 'B7': 0.87858}

    f1_50 = {'B0': 0.7284, 'Baseline(B2)': 0.7654, 'B6': 0.8095, 'B7': 0.79724}

    # Extract the data for x and y axes
    x = list(extraction_times.values())
    y = list(accuracy.values())

    # Create the scatter plot
    plt.scatter(x, y)

    # Add labels to the data points
    for i, model in enumerate(extraction_times.keys()):
        plt.annotate(model, (x[i] + 50, y[i]))

    plt.plot(x, y)
    # Set the axis labels and title
    plt.xlabel('Extraction Times')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Extraction Times')
    plt.ylim(0.6, 1)

    # Display the plot
    plt.show()

def graph_2():

    f1_10 = {'B0': 0.8113, 'Baseline(B2)': 0.8685, 'B6': 0.9202, 'B7': 0.9005}

    # Extract the data for x and y axes
    x = list(extraction_times.values())
    y = list(f1_10.values())

    # Create the scatter plot
    plt.scatter(x, y)

    # Add labels to the data points
    for i, model in enumerate(extraction_times.keys()):
        plt.annotate(model, (x[i] + 50, y[i]))

    plt.plot(x, y)
    # Set the axis labels and title
    plt.xlabel('Extraction Times')
    plt.ylabel('f1@10')
    plt.title('f1@10 vs Extraction Times')
    plt.ylim(0.6, 1)

    # Display the plot
    plt.show()


def graph_3():

    f1_25 = {'B0': 0.8015, 'Baseline(B2)': 0.86624, 'B6': 0.8926, 'B7': 0.87858}

    # Extract the data for x and y axes
    x = list(extraction_times.values())
    y = list(f1_25.values())

    # Create the scatter plot
    plt.scatter(x, y)

    # Add labels to the data points
    for i, model in enumerate(extraction_times.keys()):
        plt.annotate(model, (x[i] + 50, y[i]))

    plt.plot(x, y)
    # Set the axis labels and title
    plt.xlabel('Extraction Times')
    plt.ylabel('f1@25')
    plt.title('f1@25 vs Extraction Times')
    plt.ylim(0.6, 1)

    # Display the plot
    plt.show()


def graph_4():

    f1_50 = {'B0': 0.7284, 'Baseline(B2)': 0.7654, 'B6': 0.8095, 'B7': 0.79724}

    # Extract the data for x and y axes
    x = list(extraction_times.values())
    y = list(f1_50.values())

    # Create the scatter plot
    plt.scatter(x, y)

    # Add labels to the data points
    for i, model in enumerate(extraction_times.keys()):
        plt.annotate(model, (x[i] + 50, y[i]))

    plt.plot(x, y)
    # Set the axis labels and title
    plt.xlabel('Extraction Times')
    plt.ylabel('f1@50')
    plt.title('f1@50 vs Extraction Times')
    plt.ylim(0.6, 1)

    # Display the plot
    plt.show()


if __name__ == '__main__':
    total_graph()
