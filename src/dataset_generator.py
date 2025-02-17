import torch
import h5py
import matplotlib.pyplot as plt

def generate_path_vector(num_steps=4, image_size=32, max_step=5, start_pt=None):
    # starting point is default (0,0)
    if start_pt is None:
        start_pt = (0, 0)

    # convert starting point to tensor      
    last_p = torch.tensor([float(start_pt[0]), float(start_pt[1])])

    # store points visited in path 
    path_vector = torch.zeros(num_steps * 2, dtype=torch.float32)

    # initialize tensor to store path vector 
    pts = [last_p.clone()]

    # generate path 
    for i in range(num_steps):
        # find min and max to stay within image 
        x_min = -min(max_step, int(last_p[0]))
        x_max = min(max_step, image_size - 1 - int(last_p[0]))
        # find min and max to stay within image
        y_min = 1
        y_max = min(max_step, image_size - 1 - int(last_p[1]))

        # break out of loop 
        if x_min > x_max or y_min > y_max:
            break

        # randomly choose a step in x axis 
        dx = torch.randint(x_min, x_max + 1, (1,)).item()
        dy = torch.randint(y_min, y_max + 1, (1,)).item()

        # store in both axis in path vector
        path_vector[2*i] = dx
        path_vector[2*i + 1] = dy

        # update current position by adding step
        last_p = last_p + torch.tensor([dx, dy], dtype=torch.float32)
        pts.append(last_p.clone())

    # stack list of points into 1 tensor     
    pts = torch.stack(pts, dim=0)
    return path_vector, pts


def points_to_image(points, image_size=32):
    # create blank image tensor with 0's
    img = torch.zeros((image_size, image_size), dtype=torch.float32)

    # extract coordinate of points as int
    ix = points[:, 0].long()
    iy = points[:, 1].long()

    # create mask to ensure points are within image bounds
    valid_mask = (ix >= 0) & (ix < image_size) & (iy >= 0) & (iy < image_size)
    img[iy[valid_mask], ix[valid_mask]] = 1.0
    return img


def visualize_path(img, points, save_path=None):
    # create figure and axis for plot 
    fig, ax = plt.subplots(figsize=(4, 4))

    # display image as grayscale
    ax.imshow(img.numpy(), cmap='gray', origin='upper', extent=(0, img.shape[1], img.shape[0], 0))
    ax.plot(points[:, 0].numpy(), points[:, 1].numpy(), marker='o', color='red')

    # plot path points as red dots
    ax.set_title("Path Visualization")
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([img.shape[0], 0])
    ax.set_aspect('equal')

    # save plot to file 
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()
    plt.close(fig)

def create_dataset_h5(filename='paths.h5', num_samples=100, image_size=32):
    # open file 
    with h5py.File(filename, 'w') as f:
        # create dataset to store in image
        dset_images = f.create_dataset('images', (num_samples, image_size, image_size), dtype='f')
        dset_paths = f.create_dataset('paths', (num_samples, 2 * 4), dtype='f')
        # generate samples and store in dataset 
        for i in range(num_samples):
            # generate random path and points
            pv, pts = generate_path_vector(num_steps=4, image_size=image_size, max_step=5)
            # convert points to binary image
            img = points_to_image(pts, image_size)
            # store image and path vector in image 
            dset_images[i] = img.numpy()
            dset_paths[i] = pv.numpy()
    print(f"Finished creating {num_samples} samples in {filename}.")

def main():
    # generate sample path and its points
    pv, pts = generate_path_vector(num_steps=4, image_size=32, max_step=10)
    # print path vector and points on path 
    print("Sample path_vector (len=8):", pv)
    print("Points on path:", pts)

    # convert the points to binary image
    img = points_to_image(pts, 32)
    # visualize path on image
    visualize_path(img, pts)
    # dataset with 100 samples
    create_dataset_h5('paths.h5', num_samples=10000, image_size=32)

if __name__ == "__main__":
    main()



# check list 
# generator_path_vector (10)
# points_to_image (5)
# visualize_path (2)
# create_dataset for h5 file (4)
# main (4)     