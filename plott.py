import numpy as np
import matplotlib.pyplot as plt
 

def plot(pos,neg):
    com = ['POSITIVE','NEGATIVE']
    
    print(pos, neg)
    data = [pos,neg]
    
    
    explode = (0.0, 0.2)
    
    colors = ( "lime", "red")
    
    wp = { 'linewidth' : 1, 'edgecolor' : "black" }
    
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%".format(pct)
    
    fig, ax = plt.subplots(figsize =(10, 7))
    wedges, texts, autotexts = ax.pie(data,
                                    autopct = lambda pct: func(pct, data),
                                    explode = explode,
                                    labels = com,
                                    shadow = True,
                                    colors = colors,
                                    startangle = 90,
                                    wedgeprops = wp,
                                    textprops = dict(color ="midnightblue"))
    
    # Adding legend
    ax.legend(wedges, com,
            title ="Commments",
            loc ="center left",
            bbox_to_anchor =(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Comments Count")
    plt.savefig('static/Image/my_plot.png')

 
# show plot
# pos = 8
# neg = 2
# plot(pos,neg)
# plt.show()