import os
import jsonlines
from copy import deepcopy
from random import shuffle

dataset_path = r"C:\Users\pc\Desktop\Maxim\data\food-101\food-101\images"

all_samples = []

names = []

for idx, class_dir in enumerate(os.listdir(dataset_path)):
    print(f"{idx}: {class_dir}")
    names.append(class_dir)
    class_path = os.path.join(dataset_path, class_dir)
    for filename in os.listdir(class_path):
        sample = {"image": f"images/{class_dir}/{filename}", "label":class_dir}
        all_samples.append(deepcopy(sample))

shuffle(all_samples)


train_size = int(len(all_samples) * 0.9)

train_samples = all_samples[:train_size]
test_samples = all_samples[train_size:]

print(names)

# with jsonlines.open('train.jsonl', mode='w') as writer:
#         writer.write_all(train_samples)

# with jsonlines.open('test.jsonl', mode='w') as writer:
#         writer.write_all(test_samples)


        