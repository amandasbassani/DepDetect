import itertools
import csv

dbname = 'modma'
# combination_csv = f'./datafiles/{dbname}/tuning/combination_net.csv'
combination_csv = f'./datafiles/{dbname}/tuning/combination_hyp.csv'

# learning_rates = [0.01, 0.005]
# optimizers = ['adam', 'rmsprop']
# kernel_sizes = [3, 5]
# n_filters_conv1 = [128, 256]
# n_filters_conv2 = [128, 256]
# units_gru = [64, 128, 256]

epochs = [80, 90, 100]
batch_size = [64, 128, 256]

# parameters = [learning_rates, optimizers, kernel_sizes, n_filters_conv1, n_filters_conv2, units_gru]
parameters = [epochs, batch_size]

combinations = list(itertools.product(*parameters))
index = range(len(combinations))

with open(combination_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(['learning_rates', 'optimizers', 'kernel_sizes', 'n_filters_conv1', 'n_filters_conv2', 'units_gru'])
    writer.writerow(['epochs', 'batch_size'])
    writer.writerows(combinations)
