import numpy as np
import matplotlib.pyplot as plt
import csv

class StepValuePlotterSingle:
    def __init__(self, steps, values, smoothing_weight=0.1, file_name=None, legend=None):
        self.steps = steps
        self.values = values
        self.smoothing_weight = smoothing_weight
        self.file_name = file_name
        self.legend = legend

    def resmooth_values(self):
        smoothing_weight = self.smoothing_weight
        smoothed_values = []

        last = np.nan if len(self.values) == 0 else self.values[0]

        for value in self.values:
            if not np.isfinite(last):
                smoothed_values.append(value)
            else:
                # 1st-order IIR low-pass filter
                smoothed_value = last * smoothing_weight + (1 - smoothing_weight) * value
                smoothed_values.append(smoothed_value)
            last = smoothed_values[-1]

        return smoothed_values

    def resmooth_and_plot_single(self, ax):
        # Apply smoothing
        smoothed_values = self.resmooth_values()

        # Plotting the data on the given axis
        ax.plot(self.steps, smoothed_values, marker='', label=self.legend if self.legend else self.file_name)
        ax.set_xlabel('Step')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Overfitting')
        ax.grid(True)

class StepValuePlotter:
    def __init__(self, file_paths, legends, smoothing_weight=0.1, output_file='plot.png'):
        self.file_paths = file_paths
        self.legends = legends
        self.output_file = output_file
        self.smoothing_weight = smoothing_weight

    def read_csv(self, file_path):
        steps = []
        values = []

        with open(file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader)  # Skip the header if present
            for row in csv_reader:
                steps.append(int(row[1]))
                values.append(float(row[2]))

        return steps, values

    def resmooth_and_plot(self):
        # Plot multiple datasets on the same figure
        fig, ax = plt.subplots()

        for file_path, legend in zip(self.file_paths, self.legends):
            # Create a new object for each dataset
            steps, values = self.read_csv(file_path)
            plotter = StepValuePlotterSingle(steps, values, smoothing_weight=self.smoothing_weight, file_name=file_path.split("/")[-1], legend=legend)

            # Call the resmooth_and_plot method
            plotter.resmooth_and_plot_single(ax)

        # Add legend outside the loop
        ax.legend()

        # Save the combined figure
        plt.savefig(self.output_file)
        plt.show()

if __name__ == "__main__":
    file_paths = ["/home/vault/iwfa/iwfa048h/csv (3).csv", "/home/vault/iwfa/iwfa048h/csv (4).csv"]
    legends = ["validation", "train"]
    plotter = StepValuePlotter(file_paths, legends, smoothing_weight=0.55, output_file='combined_plot.png')
    plotter.resmooth_and_plot()
