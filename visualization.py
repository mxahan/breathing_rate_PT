import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

def get_pink_to_yellow_cmap():
    colors = [(1, 0, 0), (0, 1, 0)]  # Pink to Yellow
    cmap_name = 'pink_to_yellow'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

def image_show(rgbs, trajs_e):
    plt.figure()
    norm = Normalize(vmin=0, vmax=8)
    cmap = get_pink_to_yellow_cmap()
    for frame_no in range(rgbs.shape[1]):
        img = rgbs[0, frame_no].cpu().permute([1, 2, 0]) / 255.0
        plt.imshow(img)
    
    
        for i in range(frame_no + 1):
            x_coords = trajs_e[0, i][:, 0].cpu()
            y_coords = trajs_e[0, i][:, 1].cpu()
    
            color = cmap(norm(i))
            plt.scatter(x_coords, y_coords, color=color, marker='.', s=12)
            # if (i>0):
            #     plt.plot([prevx_coords, x_coords], [prevy_coords, y_coords], color=color, linewidth=1, alpha=0.6)
            # prevx_coords = x_coords
            # prevy_coords = y_coords
    
        plt.axis('off')  # Turn off axis labels
    
        # Add a colorbar to show the mapping between iteration and color
        # sm = ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ticks=range(frame_no + 1))
        # cbar.set_label('Iteration')
    
        plt.pause(0.1)  # Pause for 2 seconds
        
        # if frame_no == 7:
        #     break
        
        plt.clf()  # Clear the current figure for the next iteration

# Example usage
# frame_no = 5  # Change this to the desired frame number
# image_show(rgbs, trajs_e, frame_no)
