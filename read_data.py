import numpy as np
from torch.utils.data import Dataset
import torch
import read_mat_file
from read_mat_file import read_mat

data_dict = read_mat(read_mat_file.data_param)
for key in data_dict:
    print(key)

data_original_healthy = data_dict['original_healthy'][81921:163840]
data_original_or_fault = data_dict['original_or_fault'][81921:163840]
data_original_ir_fault = data_dict['original_ir_fault'][81921:163840]

data_mass_changed_healthy = data_dict['mass_changed_healthy'][81921:163840]
data_mass_changed_or_fault = data_dict['mass_changed_or_fault'][81921:163840]
data_mass_changed_ir_fault = data_dict['mass_changed_ir_fault'][81921:163840]

data_stiffness_changed_healthy = data_dict['stiffness_changed_healthy'][81921:163840]
data_stiffness_changed_or_fault = data_dict['stiffness_changed_or_fault'][81921:163840]
data_stiffness_changed_ir_fault = data_dict['stiffness_changed_ir_fault'][81921:163840]


def sample_data(signal_data, sample_length, num_samples, step_size=256):
    """
    从信号数据中按给定的长度和样本数量进行滑动窗口采样
    signal_data: 输入的信号数据
    sample_length: 每个样本的长度
    num_samples: 需要采样的样本数量
    step_size: 滑动窗口的步长，默认为1
    """
    total_length = len(signal_data)
    # print(total_length)
    samples = []
    # 计算最大可能的样本数量
    max_possible_samples = (total_length - sample_length) // step_size + 1
    num_samples = max_possible_samples

    # 确保样本数量不超过数据长度
    # if num_samples > max_possible_samples:
    # raise ValueError(f"无法从数据中采样 {num_samples} 个样本。最大可采样数量为 {max_possible_samples}。")
    # 滑动窗口采样
    for i in range(num_samples):
        start_idx = i * step_size
        samples.append(signal_data[start_idx:start_idx + sample_length])
    return np.array(samples)


# 设置采样参数
sample_length = 8192
num_samples = len

# 对每个数据进行采样
original_healthy = sample_data(data_original_healthy, sample_length, num_samples)
original_or_fault = sample_data(data_original_or_fault, sample_length, num_samples)
original_ir_fault = sample_data(data_original_ir_fault, sample_length, num_samples)

mass_changed_healthy = sample_data(data_mass_changed_healthy, sample_length, num_samples)
mass_changed_or_fault = sample_data(data_mass_changed_or_fault, sample_length, num_samples)
mass_changed_ir_fault = sample_data(data_mass_changed_ir_fault, sample_length, num_samples)

stiffness_changed_healthy = sample_data(data_stiffness_changed_healthy, sample_length, num_samples)
stiffness_changed_or_fault = sample_data(data_stiffness_changed_or_fault, sample_length, num_samples)
stiffness_changed_ir_fault = sample_data(data_stiffness_changed_ir_fault, sample_length, num_samples)

signal_types = {
    'heal': 0,
    'or': 1,
    'ir': 2,
}


def create_labels(num_samples, dict, signal_type):
    """
    根据信号类型和频率为每个样本创建标签
    num_samples: 每个类型的样本数量
    signal_type: 信号类型（wq, nq, gdt, zc）
    frequency: 采样频率（4925或7102）
    """
    labels = [dict[signal_type]] * num_samples
    return np.array(labels)


# 创建标签
# wq_4925_labels = create_labels(len(wq_4925_samples),signal_types, "wq")
# nq_4925_labels = create_labels(len(nq_4925_samples),signal_types, "nq")
# gdt_4925_labels = create_labels(len(gdt_4925_samples),signal_types, "gdt")
# zc_4925_labels = create_labels(len(zc_4925_samples),signal_types, "zc")

# 创建标签
original_healthy_labels = create_labels(len(original_healthy), signal_types, "heal")
original_or_fault_labels = create_labels(len(original_or_fault), signal_types, "or")
original_ir_fault_labels = create_labels(len(original_ir_fault), signal_types, "ir")

mass_changed_healthy_labels = create_labels(len(mass_changed_healthy), signal_types, "heal")
mass_changed_or_fault_labels = create_labels(len(mass_changed_or_fault), signal_types, "or")
mass_changed_ir_fault_labels = create_labels(len(mass_changed_ir_fault), signal_types, "ir")

stiffness_changed_healthy_labels = create_labels(len(stiffness_changed_healthy), signal_types, "heal")
stiffness_changed_or_fault_labels = create_labels(len(stiffness_changed_or_fault), signal_types, "or")
stiffness_changed_ir_fault_labels = create_labels(len(stiffness_changed_ir_fault), signal_types, "ir")


def add_noise(signal, snr_db):
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)

    # 计算噪声功率
    snr = 10 ** (snr_db / 10)  # 将信噪比（单位dB）转换为线性信噪比
    noise_power = signal_power / snr

    # 生成高斯噪声
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

    # 添加噪声后的信号
    noisy_signal = signal + noise
    return noisy_signal


class MyDataset(Dataset):
    def __init__(self, data_4925, labels, domain_label):
        self.data_4925 = data_4925
        self.labels = labels
        self.domain_label = domain_label

    def __len__(self):
        return len(self.data_4925)

    def __getitem__(self, idx):
        return self.data_4925[idx], self.labels[idx], self.domain_label[idx]


# 合并数据和标签
all_data = np.concatenate([original_healthy, original_or_fault, original_ir_fault,
                           mass_changed_healthy, mass_changed_or_fault, mass_changed_ir_fault,
                           stiffness_changed_healthy, stiffness_changed_or_fault, stiffness_changed_ir_fault])

all_labels = np.concatenate([original_healthy_labels, original_or_fault_labels, original_ir_fault_labels,
                             mass_changed_healthy_labels, mass_changed_or_fault_labels, mass_changed_ir_fault_labels,
                             stiffness_changed_healthy_labels, stiffness_changed_or_fault_labels,
                             stiffness_changed_ir_fault_labels])

# vstack here of n datasets


domain_label = ([0] * len(original_healthy_labels) * 3 +
                [1] * len(mass_changed_healthy_labels) * 3 +
                [2] * len(stiffness_changed_healthy_labels) * 3)

my_dataset = MyDataset(all_data, all_labels, domain_label)

train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=32, shuffle=True, drop_last=True)

print(my_dataset[-1])
