import requests
import json

prompt = "Write a python solution for the longest increasing path in a matrix."
data = {"prompt": prompt, "max_tokens": 512, "temperature": 0.7}

def test_endpoint(url):
    print(f"\n--- Testing {url} ---")
    response = requests.post(url, json=data, stream=True)
    text = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                payload = json.loads(decoded_line[6:])
                if 'token' in payload:
                    text += payload['token']
                    print(payload['token'], end='', flush=True)
    print()

test_endpoint("http://127.0.0.1:8000/generate/base")
test_endpoint("http://127.0.0.1:8000/generate/finetuned")
