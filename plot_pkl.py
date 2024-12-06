# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import pickle


# with open('vars.pkl', 'rb') as f:
#     reward_over_episodes = pickle.load(f)

# # Prepare the DataFrame
# df = pd.DataFrame({
#     "episodes": range(1, len(reward_over_episodes) + 1),  # x-axis: episode numbers
#     "reward": reward_over_episodes  # y-axis: reward values
# })

# # Set plot style
# sns.set_theme(style="darkgrid", context="talk", palette="rainbow")

# # Plot the lineplot
# sns.lineplot(x="episodes", y="reward", data=df).set(
#     title="REINFORCE for BankHeist"
# )

# # Show the plot
# plt.show()

import pickle

# Load the .pkl file
with open('vars.pkl', 'rb') as file:
    data = pickle.load(file)

# Print the type of the loaded object and the first few elements
print(type(data))
print(data)

