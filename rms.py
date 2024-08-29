import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

within15 = []
within10 = []
within5 = []
within1 = []

df = pd.read_csv("rmsruns.csv")
for substr in ["first k values", "last k values", "random k values"]:
    #  & df["Run"].str.contains("individual k")
    within15.append(df.loc[df["Run"].str.contains(substr), 'Within_15'].mean())
    within10.append(df.loc[df["Run"].str.contains(substr), 'Within_10'].mean())
    within5.append(df.loc[df["Run"].str.contains(substr), 'Within_5'].mean())
    within1.append(df.loc[df["Run"].str.contains(substr), 'Within_1'].mean())

# set width of bar 
barWidth = 0.2
fig = plt.subplots(figsize = (12, 8))

# Set position of bar on X axis 
br1 = np.arange(len(within15))
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 

# Make the plot
plt.bar(br1, within15, color ='r', width = barWidth, 
        edgecolor ='grey', label ='within 15%') 
plt.bar(br2, within10, color ='g', width = barWidth, 
        edgecolor ='grey', label ='within 10%') 
plt.bar(br3, within5, color ='b', width = barWidth, 
        edgecolor ='grey', label ='within 5%') 
plt.bar(br4, within1, color ='y', width = barWidth, 
        edgecolor ='grey', label ='within 1%') 

# Adding Xticks 
plt.xlabel('k value used to train with kRMSNorm', fontweight ='bold', fontsize = 15) 
plt.ylabel('Element Sampling Method', fontweight ='bold', fontsize = 15) 
plt.ylim(0, 384)
plt.title("Avg Number of Elements Needed to get with x% of Total RMS")
plt.xticks([r + 0.3 for r in range(len(within15))], 
        #    , 'k=90', 'k=100', 'k=200', 'k=300'
        ["first k values", "last k values", "random k values"])

plt.legend()
plt.show()