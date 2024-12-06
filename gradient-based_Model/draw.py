import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Load the event file
event_file = "event/events.out.tfevents.c0.1-ent0.01-lr-2.5e-4"
event_file1 ="event/events.out.tfevents.c0.1-ent0.01-lr-2.5e-3"
event_file2 ="event/events.out.tfevents.c0.3-ent0.05-lr-1e-3"
event_file3 ="event/events.out.tfevents.trial"

event_acc = EventAccumulator(event_file)
event_acc.Reload()

event_acc1 = EventAccumulator(event_file1)
event_acc1.Reload()

event_acc2 = EventAccumulator(event_file2)
event_acc2.Reload()

event_acc3 = EventAccumulator(event_file3)
event_acc3.Reload()

# Get scalar keys
scalar_tags = event_acc.Tags()["scalars"]
print("Available scalar tags:", scalar_tags)

# Retrieve data for a specific scalar
tag = scalar_tags[-2]  
scalar_data = event_acc.Scalars(tag)
scalar_data1 = event_acc1.Scalars(tag)
scalar_data2 = event_acc2.Scalars(tag)
scalar_data3 = event_acc3.Scalars(tag)

# Extract steps and values
steps = np.array([x.step for x in scalar_data1]) / 1e6  # Convert steps to millions
values = np.array([x.value for x in scalar_data])
values1 = np.array([x.value for x in scalar_data1])
values2 = np.array([x.value for x in scalar_data2])
values3 = np.array([x.value for x in scalar_data3])

# Fit a polynomial trendline (non-linear fit, e.g., degree 4)
degree = 4
coefficients = np.polyfit(steps, values, degree)
coefficients1 = np.polyfit(steps, values1, degree)
coefficients2 = np.polyfit(steps, values2, degree)
coefficients3 = np.polyfit(steps, values3, degree)
polynomial = np.poly1d(coefficients)
polynomial1 = np.poly1d(coefficients1)
polynomial2 = np.poly1d(coefficients2)
polynomial3 = np.poly1d(coefficients3)
# Generate trendline values
trendline_values = polynomial(steps)
trendline_values1 = polynomial1(steps)
trendline_values2 = polynomial2(steps)
trendline_values3 = polynomial3(steps)

# Custom formatter for the x-axis to add 'M'
def format_with_m(x, _):
    return f"{x:.0f}M"

# Plot the scalar with trendline
plt.figure(figsize=(10, 6))
plt.plot(steps, values2, label="Combo1-clip0.3-ent_coe-0.05-l_rate-1.0e-3-v_coe-0.5", linestyle="--", marker="o", alpha=0.2)
plt.plot(steps, values1, label="Combo2-clip0.1-ent_coe-0.01-l_rate-2.5e-3-v_coe-0.5", linestyle="--", marker="*", alpha=0.3)
plt.plot(steps, values, label="Combo3-clip0.1-ent_coe-0.01-l_rate-2.5e-4-v_coe-0.5", linestyle="--", marker="o", alpha=0.4)
plt.plot(steps, values3, label="Combo4-clip0.1-ent_coe-0.01-l_rate-2.5e-4-v_coe-1.0", linestyle="--", marker=".", alpha=0.2)
plt.plot(steps, trendline_values2, color="blue", label=f"Trendline-Combo1", linewidth=2)
plt.plot(steps, trendline_values1, color="orange", label=f"Trendline-Combo2", linewidth=2)
plt.plot(steps, trendline_values, color="#006400", label=f"Trendline-Combo3", linewidth=2)
plt.plot(steps, trendline_values3, color="red", label=f"Trendline-Combo4", linewidth=2)

plt.xlabel("Step (Millions, 'M')")
plt.ylabel(tag)
plt.ylim(0, 700)
plt.title(f"PPO - Episodic Return over Steps: {tag}")

# Apply custom formatter
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_with_m))

plt.legend()
plt.grid(True)
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# # Load the event file
# event_file = "event/events.out.tfevents.1733023691.YZsMac-2106.local.39850.0"
# event_acc = EventAccumulator(event_file)
# event_acc.Reload()

# # Get scalar keys
# scalar_tags = event_acc.Tags()["scalars"]
# print("Available scalar tags:", scalar_tags)

# # Retrieve data for the last two tags
# tag_1 = scalar_tags[-1]  # Most recent scalar tag
# tag_2 = scalar_tags[-2]  # Second most recent scalar tag

# # Extract data for both tags
# scalar_data_1 = event_acc.Scalars(tag_1)
# scalar_data_2 = event_acc.Scalars(tag_2)

# # Extract values for both tags (no steps required here)
# values_1 = np.array([x.value for x in scalar_data_1])
# values_2 = np.array([x.value for x in scalar_data_2])

# # Plot data of values_1 on x-axis and values_2 on y-axis
# plt.figure(figsize=(10, 6))
# plt.scatter(values_1, values_2, label=f"Data: {tag_1} vs {tag_2}", alpha=0.7)

# plt.xlabel(f"Values from {tag_1}")
# plt.ylabel(f"Values from {tag_2}")
# plt.title(f"Scatter Plot of {tag_1} vs {tag_2}")
# plt.legend()
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import FuncFormatter
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# # Load the event file
# event_file = "event/events.out.tfevents.1733268500.YZsMac-2106.local.11163.0"
# event_acc = EventAccumulator(event_file)
# event_acc.Reload()

# # Get scalar keys
# scalar_tags = event_acc.Tags()["scalars"]
# print("Available scalar tags:", scalar_tags)

# # Retrieve data for the last two tags
# tag_1 = scalar_tags[1]  # Most recent scalar tag
# tag_2 = scalar_tags[2]  # Second most recent scalar tag

# # Extract data for both tags
# scalar_data_1 = event_acc.Scalars(tag_1)
# scalar_data_2 = event_acc.Scalars(tag_2)

# # Extract steps and values for both tags
# steps_1 = np.array([x.step for x in scalar_data_1]) / 1e6  # Convert steps to millions
# values_1 = np.array([x.value for x in scalar_data_1])

# steps_2 = np.array([x.step for x in scalar_data_2]) / 1e6  # Convert steps to millions
# values_2 = np.array([x.value for x in scalar_data_2])

# # Ensure both datasets have the same number of steps (if needed, sync the steps)
# # For simplicity, we assume the same steps; otherwise, interpolation can be done.

# # Custom formatter for the x-axis to add 'M'
# def format_with_m(x, _):
#     return f"{x:.0f}M"

# # Plot data of y-axis from tag[-1] against y-axis from tag[-2]
# plt.figure(figsize=(10, 6))
# plt.plot(steps_1, values_1, label=f"{tag_1}", linestyle="--", marker="o", alpha=0.7)
# plt.plot(steps_2, values_2, label=f"{tag_2}", linestyle="--", marker="x", alpha=0.7)

# plt.xlabel("Step (Millions, 'M')")
# plt.ylabel("Values")
# plt.title(f"PPO-Clip0.-Entr_Coe-0.05")

# # Apply custom formatter
# plt.gca().xaxis.set_major_formatter(FuncFormatter(format_with_m))

# plt.legend()
# plt.grid(True)
# plt.show()
