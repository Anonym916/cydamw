import os
import pickle
import torch

archive_root = "/home/u5611943/Downloads/state_dicts/resnet50/archive"
pkl_path = os.path.join(archive_root, "data.pkl")

print("Loading VERY old TorchVision archive:", pkl_path)

# ---- 关键：定义 persistent_load ----
def persistent_load(saved_id):
    assert isinstance(saved_id, tuple)
    typename = saved_id[0]
    data = saved_id[1:]

    # Tensor storage is in ./data/xxxx files
    if typename == 'storage':
        storage_type, key, location, size = data
        filename = os.path.join(archive_root, "data", str(key))
        print("Loading tensor storage:", filename)

        storage = torch.FloatStorage.from_file(filename, False, size)
        return storage

    raise RuntimeError("Unknown persistent_load typename: " + str(typename))


# ---- 用带自定义 persistent_load 的 Unpickler ----
with open(pkl_path, "rb") as f:
    unpickler = pickle.Unpickler(f)
    unpickler.persistent_load = persistent_load
    model_data = unpickler.load()

print("Archive loaded successfully!")

# ---- 处理 state_dict ----
if "state_dict" in model_data:
    sd = model_data["state_dict"]
else:
    sd = model_data

# 删除 classifier 层（你的 SSL 只需要 backbone）
clean_sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}

# 保存新文件
save_path = "resnet50_converted.pth"
torch.save(clean_sd, save_path)

print("Saved converted model to:", save_path)
print("Number of parameters loaded:", len(clean_sd))
