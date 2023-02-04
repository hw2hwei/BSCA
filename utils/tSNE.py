import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


labl2color_dict = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (245, 130, 0),
    4: (145, 30, 180),
    5: (37, 148, 140),
    6: (128, 128, 0),
    7: (50, 228, 128),
    8: (188, 246, 12),
    9: (250, 190, 190),
    10: (0, 128, 128),
    11: (70, 240, 240),
    12: (154, 99, 36),
    13: (255, 250, 200),
    14: (128, 0, 0),
    15: (170, 255, 195),
    16: (240, 50, 230),
    17: (255, 216, 177),
    18: (0, 0, 117),
    19: (128, 128, 128),
    20: (230, 25, 75),
    21: (230, 110, 255),
    22: (215, 25, 25),
    23: (107, 9, 216),
    24: (205, 30, 49),
    25: (185, 130, 180),
    26: (110, 40, 240),
    27: (200, 150, 230),
    28: (148, 146, 12),
    29: (210, 90, 190),
    30: (50, 28, 128),
}

cls2label_dict = {
    0 : "Farmland",
    1 : "Parking",
    2 : "Industrial",
    3 : "Meadow",
    4 : "River",
    5 : "Residential",
    6 : "Forest",
}

def tSNE(data, label, title, len_s, n_classes):
    def plot_embedding(data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600 
        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(data.shape[0]):
            # label limitation
            label_i = label[i]
            label_i = label_i  if label_i < 31  else 31

            # feat_s
            if i < len_s:
                size = 4
                color = labl2color_dict[label_i]
                # color = [255,0,0]
                color =  [j/255.0 for j in color] 
                marker = 'x'
            # feat_t
            else:
                size = 4
                color = labl2color_dict[label_i]
                # color = [0,0,255]
                color =  [j/255.0 for j in color] 
                marker = '^'
           
            # plot 
            plt.scatter(data[i, 0], data[i, 1], s=size, color=(color[0], color[1], color[2]), marker=marker, linewidths=0.3)

        for i in range(0, n_classes):
            color = labl2color_dict[i]
            color =  [j/255.0 for j in color] 
            label_i = cls2label_dict[i]
            plt.plot([], [], color=color, linewidth=5, linestyle="-", label=label_i)
        plt.legend(loc='upper right', fontsize=5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(title)

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    plot_embedding(result, label, title)
    # plt.show(fig)
