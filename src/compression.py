import base64
import zlib
import json

def encode_str(s: str) -> str:
    """Compresses and encodes a string using zlib and base64.
    
    Args:
        s (str): The string to encode.
        
    Returns:
        str: The compressed and encoded string.
    """
    # Convert the string to bytes using UTF-8 encoding
    data = s.encode('utf-8')
    
    # Compress the data using zlib
    compressed_data = zlib.compress(data)
    
    # Encode the compressed data as base64 and convert to a string
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    
    return encoded_data


def decode_str(s: str) -> str:
    """Decodes and decompresses a string previously encoded using encode_str.
    
    Args:
        s (str): The string to decode.
        
    Returns:
        str: The decompressed string.
    """
    # Decode the string from base64 and convert to bytes
    encoded_data = s.encode('utf-8')
    compressed_data = base64.b64decode(encoded_data)
    
    # Decompress the data using zlib
    data = zlib.decompress(compressed_data)
    
    # Convert the data from bytes to string using UTF-8 encoding
    return data.decode('utf-8')


def encode_obj(obj):
    obj_json = json.dumps(obj)
    return encode_str(obj_json)

def decode_obj(obj):
    obj_json = decode_str(obj)
    return json.loads(obj_json)