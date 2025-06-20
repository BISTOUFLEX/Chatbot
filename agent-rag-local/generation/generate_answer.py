"""Generate answers using a local Ollama model."""

import subprocess


class AnswerGenerator:
    def __init__(self, model_name: str = "llama2") -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        proc = subprocess.run(["ollama", "run", self.model_name], input=prompt.encode(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode())
        return proc.stdout.decode()


if __name__ == "__main__":
    print(AnswerGenerator().generate("Bonjour"))
