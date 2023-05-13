import logging
logging.basicConfig(level=logging.INFO)
from src.md_utils import fix_lines

tmp = """| Language | Popularity |
| --- | --- |
| Python | 3rd |
| Java | 2nd |
| C++ | 2nd (C++11) |
| C# | 2nd (.NET) |
| JavaScript | 2nd |
| Ruby | 1st |
| PHP | 1st |
| Swift | 1st (Apple) |
| Go | 1st |
| Rust | 1st |"""

tests = [tmp,
"""Create a docstring for the below python function
def get_args():                                                
    parser = argparse.ArgumentParser(description='Get model choice and token')
    parser.add_argument('--model', default='models/GLaDOSRedPajama7BBase', help='Path to the model to run')
    parser.add_argument('--token', default=None, help='Huggingface token required for starcoder model download')
    parser.add_argument('--multi_gpu', action="store_true", default=False, help='If passed will distribute model across multip
le GPUs')              
    args = parser.parse_args()                                 
    return args                                                
GLaDOS :
Here's a docstring for your Python function:    

```                                                            
python                                                         
def get_arguments():     
    # Parse arguments    
    parser = ArgumentParser()
    # Add arguments to parser                                  
    parser.description = "Get model choice and required token"
    # Add optional arguments                                   
    parser_add_argument("--model", default=None,
                        help="Path to model to run")
    # Add required argument
    parser_required_argument("--token", default=None)
    # Add multi-gpu option
        parser.description = "Get model choice and required token"                                                         [26/74]
    # Add optional arguments                                                                                                  
    parser_add_argument("--model", default=None,                                                                              
                        help="Path to model to run")                                                                          
    # Add required argument                                                                                                   
    parser_required_argument("--token", default=None)                                                                         
    # Add multi-gpu option                                                                                                    
    parser_optional_argument("--multi_gpu", default=False)                                                                    
    # Return parsed arguments                                                                                                 
    return parser                                              
                                                                                                                              
```  
"""

]

for tmp in tests:
    print(fix_lines(tmp))
