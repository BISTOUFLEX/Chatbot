import subprocess

MODEL_NAME = "llama2"


def generate(prompt: str) -> str:
    proc = subprocess.run(["ollama", "run", MODEL_NAME], input=prompt.encode(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    return proc.stdout.decode()


if __name__ == "__main__":
    print(generate("Bonjour"))
