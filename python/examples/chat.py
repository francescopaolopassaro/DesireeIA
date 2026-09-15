"""DesireeIA Python wrapper — end-to-end example.

Runs a chat conversation and streams the reply token by token.

Usage:
    python examples/chat.py path/to/model.gguf [prompt]
"""

import sys

import desireeia


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Raccontami una barzelletta breve."

    print(f"DesireeIA engine: {desireeia.version()}")
    hw = desireeia.detect_hardware()
    print(
        f"HW: {hw.cpu_threads} threads, "
        f"{'AVX2 ' if hw.avx2 else ''}{'CUDA ' if hw.cuda_device_count else ''}"
        f"{'Metal ' if hw.metal else ''}{'Vulkan ' if hw.vulkan else ''}"
    )

    plan = desireeia.build_plan(model_path)
    print(f"Plan: {plan}")

    with desireeia.LocalModel.load(model_path, plan) as model:
        print(f"\nModel loaded: {model.model_path}\n")

        # 1. Sampling: a touch of temperature for variety
        model.set_sampling(
            desireeia.SamplingOptions(
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                penalty_repeat=1.1,
            )
        )

        # 2. Streaming chat generation with a stop sequence
        options = desireeia.GenerateOptions(
            max_tokens=256,
            stop_sequences=["</s>", "<|im_end|>"],
        )
        messages = [("system", "Sei un assistente disponibile e conciso."),
                    ("user", prompt)]

        print("Assistant: ", end="", flush=True)
        for piece in model.chat_stream(messages, options):
            print(piece, end="", flush=True)
        print("\n")

        # 3. Show the raw tokens for the prompt
        ids = model.tokenize(prompt)
        eos = model.eos_id
        print(f"Prompt tokenized into {len(ids)} tokens (EOS id = {eos})")

    return 0


if __name__ == "__main__":
    sys.exit(main())