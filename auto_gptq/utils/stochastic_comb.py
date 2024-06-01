import torch

def stochastically_combine_tensors(tensor1, tensor2, prob1, prob2):
    """
    Stochastically combine two tensors based on given probabilities.
    
    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.
        prob1 (float): The probability of selecting elements from tensor1.
        prob2 (float): The probability of selecting elements from tensor2.
    
    Returns:
        torch.Tensor: The resulting tensor after stochastic combination.
    """
    # Ensure that probabilities sum to 1
    assert prob1 + prob2 == 1.0, "Probabilities must sum to 1"
    
    # Generate a random matrix with values between 0 and 1
    random_matrix = torch.rand(tensor1.shape).to("cuda")
    
    # Create the resulting tensor by selecting elements from tensor1 or tensor2 based on the probabilities
    result_tensor = torch.where(random_matrix < prob1, tensor1, tensor2)
    return result_tensor.to("cuda")