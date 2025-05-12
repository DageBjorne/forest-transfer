import numpy as np


def random_rotation(raster_image):
    new_raster_image = np.copy(raster_image)
    k = np.random.randint(0,4, size = 1)
    new_raster_image = np.rot90(new_raster_image, k = k)
    return new_raster_image

def mirroring(raster_image):
    new_raster_image = np.copy(raster_image)
    new_raster_image = np.fliplr(new_raster_image)
    return new_raster_image

def swap_neighboring_pixels(raster_image):
    
    j, k = np.random.randint(1, 7, size = 1)[0], np.random.randint(1, 7, size = 1)[0]
    j_increment = np.random.randint(-1, 2, size = 1)[0]
    j_neighbor = j + j_increment
    k_increment = np.random.randint(-1, 2, size = 1)[0]
    k_neighbor = k + k_increment
        
    jk_pixel_values = raster_image[j,k,:]
    jk_neighbor_pixel_values = raster_image[j_neighbor, k_neighbor,:]
    raster_image[j,k,:] = jk_neighbor_pixel_values
    raster_image[j_neighbor, k_neighbor,:] = jk_pixel_values   
    return raster_image

def swap_pixels(raster_image, swaps):
    # Perform multiple swaps in one function call
    new_raster_image = np.copy(raster_image)
    for j, k, j_, k_ in swaps:
        jk_pixel_values = new_raster_image[j, k, :]
        j_k__pixel_values = new_raster_image[j_, k_, :]
        new_raster_image[j, k, :] = j_k__pixel_values
        new_raster_image[j_, k_, :] = jk_pixel_values
    return new_raster_image

def augment_raster_image(raster_image, include_swapping,
                         swapping_range, rotation_prob, fliplr_prob):
    
    new_raster_image = np.transpose(raster_image, (1, 2, 0))
    
    # Perform rotation with given probability
    if np.random.uniform(0, 1) < rotation_prob:
        new_raster_image = random_rotation(new_raster_image)
    
    # Perform flipping with given probability
    if np.random.uniform(0, 1) < fliplr_prob:
        new_raster_image = mirroring(new_raster_image)
    
    # Perform swapping if enabled
    if include_swapping:
        # Determine number of swaps based on the given range
        nr_of_swaps = np.random.randint(swapping_range[0], swapping_range[1])
        
        # Swap neighboring pixels based on number of swaps
        for _ in range(nr_of_swaps):
            new_raster_image = swap_neighboring_pixels(new_raster_image)

    # Transpose back to original shape
    new_raster_image = np.transpose(new_raster_image, (2, 0, 1))
    
    return new_raster_image
