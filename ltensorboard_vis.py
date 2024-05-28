import numpy as np
import matplotlib.pyplot as plt
import csv
import os





class StepValuePlotter:
    def __init__(self, file_paths, output_file='plot.png', smoothing_weight=0.1):
        self.file_paths = file_paths
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

        for file_path in self.file_paths:
            # Create a new object for each dataset
            steps, values = self.read_csv(file_path)
            file_name = file_path.split("/")[-1]
            plotter = StepValuePlotterSingle(steps, values, smoothing_weight=self.smoothing_weight, file_name=file_name)

            # Call the resmooth_and_plot method
            plotter.resmooth_and_plot_single(ax)

        # Add legend outside the loop
        ax.legend()

        # Save the combined figure
        plt.savefig(self.output_file)
        plt.show()








class StepValuePlotterSingle:
    def __init__(self, steps, values, smoothing_weight=0.1, file_name=None):
        self.steps = steps
        self.values = values
        self.smoothing_weight = smoothing_weight
        self.file_name = file_name

    def resmooth_values(self, values):
        smoothing_weight = self.smoothing_weight
        smoothed_values = []

        last = np.nan if len(values) == 0 else values[0]

        for value in values:
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
        smoothed_values = self.resmooth_values(self.values)

        # Plotting the data on the given axis
        ax.plot(self.steps, smoothed_values, marker='', label=self.file_name)
        ax.set_xlabel('Step')
        ax.set_ylabel('validation loss')
        ax.set_title('Step vs Smoothed Value Plot')
        ax.grid(True)

if __name__ == "__main__":
    # file_path = "heavy_aug_550550__trial9__WH_ardataloader_val_min (1).csv"
    # plotter = StepValuePlotterSingle(file_path, output_file='smoothed_plot.png')
    # plotter.resmooth_and_plot()
    #ROOT_DIR = "/home/vault/iwfa/iwfa048h/csv_for_plotting"
    file_paths = ["/home/vault/iwfa/iwfa048h/csv.csv", "/home/vault/iwfa/iwfa048h/csv (1).csv"]
    legend = ["validation", ]
    #file_paths = ["/home/vault/iwfa/iwfa048h/l_aug_trial89_SMP_NLMLP (1).csv", "/home/vault/iwfa/iwfa048h/l_aug_trial89_SMP_NLMLP.csv"]
    #file_paths = ["/home/vault/iwfa/iwfa048h/patience5__trial9__WH_ardataloader_val_min.csv", "/home/vault/iwfa/iwfa048h/patience5__trial9__WH_ardataloader_val_min (1).csv"]
    #file_paths = ["/home/vault/iwfa/iwfa048h/patience5__trial9__WH_ardataloader_val_min (1).csv", "/home/vault/iwfa/iwfa048h/patience5__trial9__WH_ardataloader_val_min.csv"]
    plotter = StepValuePlotter(file_paths, output_file='combined_plot.png', smoothing_weight=0.75)
    plotter.resmooth_and_plot()



# class StepValuePlotterSingle:
#     def __init__(self, file_path, output_file='plot.png', smoothing_weight=0.90):
#         self.file_path = file_path
#         self.output_file = output_file
#         self.smoothing_weight = smoothing_weight

#     def read_csv(self):
#         steps = []
#         values = []

#         with open(self.file_path, 'r') as csvfile:
#             csv_reader = csv.reader(csvfile, delimiter=',')
#             next(csv_reader)  # Skip the header if present
#             for row in csv_reader:
#                 steps.append(int(row[1]))
#                 values.append(float(row[2]))

#         return steps, values


    # def resmooth_and_plot_single(self):
    #     # Read data from CSV
    #     steps, values = self.read_csv()

    #     # Apply smoothing
    #     smoothed_values = self.resmooth_values(values)

    #     # Plotting the data
    #     plt.plot(steps, smoothed_values, marker='o')
    #     plt.xlabel('Step')
    #     plt.ylabel('Smoothed Value')
    #     plt.title('Step vs Smoothed Value Plot')
    #     plt.grid(True)

    #     # Save the figure
    #     plt.savefig(self.output_file)

    # def resmooth_values(self, values):
    #     smoothing_weight = self.smoothing_weight
    #     smoothed_values = []

    #     last = np.nan if len(values) == 0 else values[0]

    #     for value in values:
    #         if not np.isfinite(last):
    #             smoothed_values.append(value)
    #         else:
    #             # 1st-order IIR low-pass filter
    #             smoothed_value = last * smoothing_weight + (1 - smoothing_weight) * value
    #             smoothed_values.append(smoothed_value)
    #         last = smoothed_values[-1]

    #     return smoothed_values


