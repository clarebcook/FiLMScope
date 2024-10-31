import torch

# make the base grid that wee need to use F.grid_sample
def generate_base_grid(image_shape):
    y, x = torch.meshgrid(
        torch.arange(0, image_shape[0], dtype=torch.float32) / image_shape[0],
        torch.arange(0, image_shape[1], dtype=torch.float32) / image_shape[1],
    )
    base_grid = torch.stack([x, y], dim=-1)
    base_grid = base_grid * 2 - 1

    return base_grid[None]

