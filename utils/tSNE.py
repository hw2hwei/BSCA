import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


labl2color_dict = {
    0: (255, 255, 255), 
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (67, 99, 216),
    4: (245, 130, 49),
    5: (145, 30, 180),
    6: (70, 240, 240),
    7: (240, 50, 230),
    8: (188, 246, 12),
    9: (250, 190, 190),
    10: (0, 128, 128),
    11: (230, 190, 255),
    12: (154, 99, 36),
    13: (255, 250, 200),
    14: (128, 0, 0),
    15: (170, 255, 195),
    16: (128, 128, 0),
    17: (255, 216, 177),
    18: (0, 0, 117),
    19: (128, 128, 128),
    20: (230, 25, 75),
    21: (20, 80, 75),
    22: (215, 25, 25),
    23: (107, 9, 216),
    24: (205, 30, 49),
    25: (185, 130, 180),
    26: (110, 40, 240),
    27: (200, 150, 230),
    28: (148, 146, 12),
    29: (210, 90, 190),
    30: (50, 28, 128),
    31: (50, 228, 128),
}


def tSNE(data, labl, title, len_s, n_classes):
    def plot_embedding(data, labl, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600 
        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(data.shape[0]):
            # label limitation
            labl_i = labl[i]
            labl_i = labl_i  if labl_i < 31  else 31

            # feat_s
            if i < len_s:
                size = 2
                color = labl2color_dict[labl_i]
                # color = labl2color_dict[1]
                color =  [j/255.0 for j in color] 
                marker = '^'
            # feat_t
            else:
                size = 2.5
                color = labl2color_dict[labl_i]
                # color = labl2color_dict[2]
                color =  [j/255.0 for j in color] 
                marker = 'x'
           
            # plot 
            plt.scatter(data[i, 0], data[i, 1], s=size, color=(color[0], color[1], color[2]), marker=marker, linewidths=0.3)

        # for i in range(0, n_classes):
        #     color = labl2color_dict[i+1]
        #     color =  [j/255.0 for j in color] 
        #     labl = houston_cls_labl[str(i+1)]
        #     plt.plot([], [], color=color, linewidth=5, linestyle="-", labl=labl)
        # plt.legend(loc='upper right', fontsize=5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(title)

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    plot_embedding(result, labl, title)
    # plt.show(fig)
