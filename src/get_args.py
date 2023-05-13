import argparse

def get_args():
    """Get arguments for running GLaDOS

    Returns:
        NameSpace: args object with member variables for each option
    """
    parser = argparse.ArgumentParser(description='Get model choice and token')
    parser.add_argument('--model', default='models/glados_redpajama7b_base', help='Path to the model to run')
    parser.add_argument('--token', default=None, help='Huggingface token required for starcoder model download')
    parser.add_argument('--multi_gpu', action="store_true", default=False, help='If passed will distribute model across multiple GPUs')
    args = parser.parse_args()
    return args