
import pickle
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def check(path):
    print(f"Checking {path} ({os.path.getsize(path)} bytes)...")
    try:
        with open(path, 'rb') as f:
            # Try loading just the first bit or full?
            # Full load 15GB might be slow and OOM.
            # But "Ran out of input" suggests EOF.
            # Maybe we can just read the end?
            # Pickle format ends with a STOP opcode (usually . or \x2e)
            f.seek(-1, 2) # Go to end
            last = f.read(1)
            print(f"Last byte: {last}")
            if last == b'.':
                print("Ends with STOP opcode (valid-ish).")
            else:
                print("Does NOT end with STOP opcode (truncated).")
            
    except Exception as e:
        print(f"Error: {e}")

paths = [
    os.path.join(BASE, "checkpoints_async", "replay_buffer.pkl"),
    os.path.join(BASE, "checkpoints", "replay_buffer.pkl"),
]
for p in paths:
    if os.path.exists(p):
        check(p)
    else:
        print(f"Skipped (not found): {p}")
