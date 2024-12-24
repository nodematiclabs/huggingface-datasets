from datasets import load_dataset

ds = load_dataset("nebius/SWE-bench-extra")

counts = {}
for license in ds["train"]["license"]:
    if license in counts.keys():
        counts[license] += 1
    else:
        counts[license] = 1

print(counts)