from utils import load_driving_log, plot_steering_histogram, balance_data

# Update this path to your actual CSV location
data = load_driving_log(r"C:\Users\kumud\OneDrive\Documents\self_driving\driving_log.csv")

print(f"Total rows: {len(data)}")

# See distribution BEFORE balancing
plot_steering_histogram(data, num_bins=25, samples_per_bin=100)

# Balance it (trims the big 0-spike)
balanced_data = balance_data(data, num_bins=25, samples_per_bin=100)

# See distribution AFTER balancing
plot_steering_histogram(balanced_data, num_bins=25, samples_per_bin=100)
