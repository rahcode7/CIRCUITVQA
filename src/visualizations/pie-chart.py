

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df= pd.DataFrame({'x':['Counting', 'Complex Counting', 'Junction','Position','Value'],'y':[20.2,0.9,57.6,18.4,2.9]})


plt.figure(figsize=(8, 8)) # change the size of a figure# The slices will be ordered and plotted counter-clockwise.
labels = df['x']
sizes = df['y']
colors = ['#FF7F50','#E84D5F','#FFB600', '#09A0DA','#8464a0'] #define colors of three donut pieces
explode = (0, 0, 0,0,0) # explode a slice if required
textprops = {'fontsize':14,'color':'black'} # Font size of text in donut 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%.2f%%', # Show data in 0.00%
 pctdistance =0.9,
 shadow=False,
 textprops =textprops,
 wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})

#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.65,color='grey', fc='white',linewidth=1.00)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal') # Set aspect ratio to be equal so that pie is      drawn as a circle.

plt.savefig('pie-chart.png')