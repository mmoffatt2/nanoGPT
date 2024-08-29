import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

within15 = []
within10 = []
within5 = []
within1 = []

df = pd.read_csv("lastrmsruns.csv")
#  'where k = 90', 'where k = 100', 'where k = 200', 'where k = 300'
for substr in [" RMSNorm ", "where k = 12",'where k = 15', 'where k = 20', 'where k = 25', 'where k = 30', 'where k = 35', 'where k = 40', 'where k = 50', 'where k = 60', 'where k = 70', 'where k = 80', "where k = 384"]:
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
plt.ylabel('Number of Elements Required to Reach "%"', fontweight ='bold', fontsize = 15) 
plt.ylim(0, 384)
plt.title("Avg Number of Elements Needed to get with x% of Total RMS")
plt.xticks([r + 0.3 for r in range(len(within15))], 
        #    , 'k=90', 'k=100', 'k=200', 'k=300'
        ['RMS', 'k=12', 'k=15', 'k=20', 'k=25', 'k=30', 'k=35', 'k=40', 'k=50', 'k=60', 'k=70', 'k=80', "k=384"])

plt.legend()
plt.show()