import torch

from loguru import logger


def threshold_li(
    image: torch.Tensor,
    initial_guess=None,
    tolerance=1e-20,
    max_iter=100,
):
    """Return the threshold value based on the Li algorithm.

    Parameters:
    image : torch.Tensor
        Input image.
    initial_guess : float, optional
        Initial threshold value. If None, use the mean of the image.
    tolerance : float, optional
        Convergence criterion. Default is 1e-20.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns:
    float
        Threshold value.
    """
    # Ensure the image is a float tensor
    image = image.to(torch.float32)

    # Initial threshold estimate
    if initial_guess is not None:
        t = torch.tensor(initial_guess, device=image.device, dtype=torch.float32)
    else:
        t = image.mean().clone().detach()

    for i in range(max_iter):
        new_t = threshold_li_step(image, t)

        # Check for convergence
        if torch.abs(new_t - t) < tolerance:
            logger.debug(
                f"Converged after {i + 1} iterations. The threshold is {new_t.item()}",
            )
            break

        t = new_t

    return t


@torch.jit.script
def threshold_li_step(image: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Perform one step of the Li thresholding algorithm.

    Parameters:
    image : torch.Tensor
        Input image.
    t : torch.Tensor
        Current threshold value.

    Returns:
    torch.Tensor
        New threshold estimate.
    """
    # Compute masks for background and object
    background_mask = image <= t
    object_mask = ~background_mask

    # Compute the means of the background and object regions
    mean_background = torch.mean(image[background_mask])
    mean_object = torch.mean(image[object_mask])

    # New threshold estimate
    new_t = (mean_background - mean_object) / (
        torch.log(mean_background) - torch.log(mean_object)
    )

    return new_t
