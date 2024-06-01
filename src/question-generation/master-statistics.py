import os 
from pathlib import Path
from icecream import ic 
import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

DATA_PATH = "datasets/questions/"


if __name__ == "__main__":

    master_df = pd.read_csv(DATA_PATH+"/all/"+"master.csv")
    ic(master_df.columns)
    ic(master_df['splittype'].value_counts())
    ic(master_df.shape)

    ic(master_df.groupby(['splittype'])['file'].nunique())
    ic(master_df['file'].nunique())



    #ic(master_df.groupby(['splittype','qtype']).size())

    #ic(master_df[['qtype','answer']].value_counts())
    #ic(master_df['file'].nunique())

    sub_df = master_df[(master_df['splittype']=='train')]   #(master_df['qtype']=='value')

    #sub_df["answer"] = pd.to_numeric(sub_df["answer"])
    ic(sub_df['answer'].value_counts())


    # hplot = sns.histplot(data=sub_df, x="answer")
    # fig = hplot.get_figure()
    # fig.savefig("hplot.png") 

    # ax = sub_df.hist(column='answer',by='splittype', bins=50, grid=False, figsize=(14,10),xrot=90,color='#86bf91', zorder=2, rwidth=0.9)
    # ax = ax[0]
    # for x in ax:

    #     # Despine
    #     # x.spines['right'].set_visible(False)
    #     # x.spines['top'].set_visible(False)
    #     # x.spines['left'].set_visible(False)

    #     #Switch off ticks
    #     x.tick_params() #axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    #     # Draw horizontal axis lines
    #     vals = x.get_yticks()
    #     for tick in vals:
    #         x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    #     # Remove title
    #     x.set_title("")

    #     # Set x-axis label
    #     x.set_xlabel("Answers of counting question", labelpad=20, weight='bold', size=12)

    #     # Set y-axis label
    #     x.set_ylabel("Number of questions", labelpad=20, weight='bold', size=12)

    #     # Format y-axis label
    #     #x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    #fig = ax.get_figure()
    plt.savefig("hplot.png") 