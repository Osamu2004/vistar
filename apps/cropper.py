import os
import sys
from tqdm import tqdm
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from skimage import io


class DatasetCropper:
    """
    根据根路径和裁剪配置对数据集进行裁剪，同时生成以裁剪参数和填充信息为后缀的子文件夹结构。
    """

    def __init__(self, root, data_crop, num_threads=4):
        """
        初始化 DatasetCropper。

        Args:
            root (str): 数据集根路径。
            data_crop (dict): 包含各子集裁剪配置的字典。
                每个子集需要包含：
                - crop_size (tuple): 裁剪图像的尺寸 (height, width)。
                - step_size (tuple): 滑动窗口的步长 (step_y, step_x)。
                - enable (bool): 是否启用裁剪。
                - padding (int): 填充大小（可选，默认为0）。
            num_threads (int): 用于多线程裁剪的线程数量。
        """
        self.root = root
        self.data_crop = data_crop
        self.num_threads = num_threads

    def validate_config(self, subset_name):
        """
        验证裁剪配置是否完整。

        Args:
            subset_name (str): 子集名称，例如 "train"、"test"。

        Raises:
            ValueError: 如果配置缺少必要参数。
        """
        subset_config = self.data_crop.get(subset_name, {})

        # 如果 enable 为 False，可以跳过其他参数
        if not subset_config.get("enable", False):
            return

        # 如果 enable 为 True，检查所有必需参数
        required_keys = ["crop_size", "step_size", "enable"]
        for key in required_keys:
            if key not in subset_config:
                raise ValueError(f"Missing '{key}' in configuration for subset '{subset_name}'.")

    def pad_image(self, image, padding):
        """
        对图像进行边缘填充。

        Args:
            image (PIL.Image.Image): 输入图像。
            padding (int): 填充大小。

        Returns:
            PIL.Image.Image: 填充后的图像。
        """
        if padding > 0:
            return ImageOps.expand(image, border=padding, fill=0)  # 默认填充黑色
        return image

    def validate_image_size(self, image_size, crop_size, step_size, padding):
        """
        验证图像大小是否能够正确裁剪。

        Args:
            image_size (tuple): 原始图像大小 (height, width)。
            crop_size (tuple): 裁剪块的大小 (height, width)。
            step_size (tuple): 滑动窗口的步长 (step_height, step_width)。
            padding (int): 填充大小。

        Raises:
            ValueError: 如果图像大小无法被裁剪。
        """
        h, w = image_size
        crop_h, crop_w = crop_size
        step_h, step_w = step_size
        h_padded, w_padded = h + 2 * padding, w + 2 * padding

        if ((h_padded - crop_h) % step_h != 0) or ((w_padded - crop_w) % step_w != 0):
            raise ValueError(
                f"Error: Image size {h}x{w} (after padding {h_padded}x{w_padded}) cannot be evenly cropped "
                f"with crop size {crop_h}x{crop_w} and step size {step_h}x{step_w}. "
                f"Ensure the padding, crop size, and step size are compatible."
            )

    def generate_output_base_path(self, subset_name):
        """
        根据裁剪配置生成数据集的裁剪结果的基础路径。

        Args:
            subset_name (str): 子集名称，例如 "train"、"test"。

        Returns:
            str: 裁剪后存储的基础目录路径。
        """
        subset_config = self.data_crop.get(subset_name, {})
        input_dir = os.path.join(self.root, subset_name)

        # 如果裁剪未启用，返回原始路径
        if not subset_config.get("enable", False):
            return input_dir

        crop_size = subset_config["crop_size"]
        step_size = subset_config["step_size"]
        padding = subset_config.get("padding", 0)

        # 根据裁剪参数生成唯一路径
        output_base_dir = f"{input_dir}_cropped_{crop_size[0]}x{crop_size[1]}_step{step_size[0]}x{step_size[1]}_pad{padding}"
        return output_base_dir

    def detect_subfolders(self, subset_name):
        """
        自动检测数据集子文件夹（如 "t1", "t2", "label"）。

        Args:
            subset_name (str): 子集名称，例如 "train"。

        Returns:
            list: 检测到的子文件夹名称列表。
        """
        subset_dir = os.path.join(self.root, subset_name)
        if not os.path.exists(subset_dir):
            raise ValueError(f"Subset directory '{subset_dir}' does not exist.")

        return [
            subfolder for subfolder in os.listdir(subset_dir)
            if os.path.isdir(os.path.join(subset_dir, subfolder))
        ]
    def crop_image_numpy(self, image, crop_size, step_size):
        """
        使用 NumPy 裁剪图像。

        Args:
            image (PIL.Image.Image): 输入图像。
            crop_size (tuple): 裁剪图像的尺寸。
            step_size (tuple): 滑动窗口的步长。

        Returns:
            list: 裁剪后的图像块和对应的 (row_idx, col_idx) 坐标。
        """
        np_image = np.array(image)
        crop_h, crop_w = crop_size
        step_y, step_x = step_size
        h, w = np_image.shape[:2]

        cropped_patches = []
        for top in range(0, h - crop_h + 1, step_y):
            for left in range(0, w - crop_w + 1, step_x):
                patch = np_image[top:top + crop_h, left:left + crop_w]
                cropped_patches.append((patch, top // step_y, left // step_x))

        return cropped_patches

    def crop_subset(self, subset_name):
        """
        对指定数据集子集进行裁剪。

        Args:
            subset_name (str): 子集名称，例如 "train"、"test"。

        Returns:
            dict: 裁剪后路径的子集名映射。
        """
        self.validate_config(subset_name)
        subset_config = self.data_crop.get(subset_name, {})

        subfolders = self.detect_subfolders(subset_name)
        subset_mapping = {}

        if not subset_config.get("enable", False):
            print(f"Skipping {subset_name} as cropping is disabled.")
            for subfolder in subfolders:
                input_dir = os.path.join(self.root, subset_name, subfolder)
                subset_mapping[subfolder] = input_dir
            return subset_mapping

        crop_size = subset_config["crop_size"]
        step_size = subset_config["step_size"]
        padding = subset_config.get("padding", 0)

        output_base_dir = self.generate_output_base_path(subset_name)

        for subfolder in subfolders:
            input_dir = os.path.join(self.root, subset_name, subfolder)
            output_dir = os.path.join(output_base_dir, subfolder)

            if os.path.exists(output_dir):
                print(f"Skipping {subset_name}/{subfolder} as output directory already exists: {output_dir}")
                subset_mapping[subfolder] = output_dir
                continue

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for root, _, files in os.walk(input_dir):
                    for file in tqdm(files, desc=f"Cropping {subset_name}/{subfolder}"):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff' , '.tif')):
                            image_path = os.path.join(root, file)
                            try:
                                image = Image.open(image_path)
                                self.validate_image_size(image.size, crop_size, step_size, padding)
                                image_padded = self.pad_image(image, padding)
                                filename, file_extension = os.path.splitext(file)
                                patches = self.crop_image_numpy(image_padded, crop_size, step_size)
                                futures.append(
                                    executor.submit(
                                        self.save_cropped_patches,
                                        patches,
                                        filename,
                                        output_dir,
                                        file_extension,
                                    )
                                )
                            except ValueError as e:
                                print(f"Validation error for {image_path}: {e}")
                                sys.exit(1)  # 直接退出程序
                            except Exception as e:
                                print(f"Error processing {image_path}: {e}")

                # 等待所有任务完成
                for future in futures:
                    future.result()

            print(f"{subset_name.capitalize()}/{subfolder} dataset cropping completed. Output stored at {output_dir}")
            subset_mapping[subfolder] = output_dir

        return subset_mapping
    def save_cropped_patches(self, patches, filename, output_dir, file_extension):
        """
        保存裁剪后的图像块。

        Args:
            patches (list): 裁剪后的图像块和对应坐标。
            filename (str): 原始图像的文件名（不含扩展名）。
            output_dir (str): 裁剪结果保存的目录。
            file_extension (str): 原始图像的文件扩展名（例如 '.jpg'）。
        """
        os.makedirs(output_dir, exist_ok=True)
        for patch, row_idx, col_idx in patches:
            patch_image = Image.fromarray(patch)
            output_path = os.path.join(output_dir, f"{filename}_{row_idx}_{col_idx}{file_extension}")
            patch_image.save(output_path)
    def crop_all(self):
        """
        对配置中的所有子集进行裁剪，并返回原始 keys 和裁剪后 keys 的映射。

        Returns:
            dict: 包含原始子集名称到裁剪后路径的映射。
        """
        overall_mapping = {}

        for subset_name in self.data_crop.keys():
            subset_mapping = self.crop_subset(subset_name)
            overall_mapping[subset_name] = subset_mapping

        return overall_mapping
    
from apps.utils.misc import dump_config, load_config, SafeLoaderWithTuple
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
class ImgStats:
    """
    用于计算指定目录下所有图像的均值和标准差，并存储到 YAML 文件中。
    """

    def __init__(self, num_threads=4):
        """
        初始化类。

        Args:
            num_threads (int): 用于并行处理图像的线程数量，默认 4。
        """
        self.num_threads = num_threads

    def _process_image(self, image_path):
        """
        处理单张图像，计算像素总和和平方和。

        Args:
            image_path (str): 图像路径。

        Returns:
            tuple: 包含像素总和、像素平方和以及图像像素数。
        """
        image = io.imread(image_path)  # 确保图像是 RGB 格式
        image_array = np.array(image) / 255.0  # 归一化到 [0, 1] 范围
        pixel_sum = image_array.sum(axis=(0, 1))
        pixel_squared_sum = (image_array ** 2).sum(axis=(0, 1))
        total_pixels = image_array.shape[0] * image_array.shape[1]
        return pixel_sum, pixel_squared_sum, total_pixels

    def _compute_mean_std(self, root_dir):
        """
        遍历目录下所有图像，计算均值和标准差。

        Args:
            root_dir (str): 图像目录的路径。

        Returns:
            dict: 包含整体均值和标准差的字典：
                {
                    "mean": (mean_c1, mean_c2, ..., mean_cn),
                    "std": (std_c1, std_c2, ..., std_cn)
                }
        """
        pixel_sum = None  # 用于累加所有通道的像素值
        pixel_squared_sum = None  # 用于累加所有通道的像素平方值
        total_pixels = 0  # 总像素数

        # 收集所有图像路径
        image_paths = []
        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff",".tif"}  # 支持的扩展名
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in supported_extensions:
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            raise ValueError(f"No supported images found in the directory: {root_dir}")

        # 并行处理图像
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(tqdm(
                executor.map(self._process_image, image_paths),
                total=len(image_paths),
                desc="Processing images"
            ))

        # 聚合结果
        for pixel_sum_part, pixel_squared_sum_part, total_pixels_part in results:
            if pixel_sum is None:  # 第一次初始化
                if isinstance(pixel_sum_part, np.ndarray):  # 处理多通道图像
                    pixel_sum = np.zeros(len(pixel_sum_part))  # 根据图像通道数初始化
                    pixel_squared_sum = np.zeros(len(pixel_squared_sum_part))  # 根据图像通道数初始化
                else:  # 处理单通道图像
                    pixel_sum = 0  # 初始化为0
                    pixel_squared_sum = 0  # 初始化为0
            pixel_sum += pixel_sum_part
            pixel_squared_sum += pixel_squared_sum_part
            total_pixels += total_pixels_part

        # 计算均值和标准差
        mean = (pixel_sum / total_pixels).round(3).tolist()  # 保留三位小数并转为 Python 列表
        std = (np.sqrt(pixel_squared_sum / total_pixels - (pixel_sum / total_pixels) ** 2)).round(3).tolist()

        return {"mean": mean, "std": std}

    def compute_mean_std(self, root_dir,output_dir):
        """
        计算图像均值和标准差，存储到 root_dir 中的 statistics.yaml 文件。如果文件已存在，则直接加载文件数据。

        Args:
            root_dir (str): 图像目录的路径。

        Returns:
            dict: 包含均值和标准差的字典。
        """

        # 如果文件已存在，直接加载并返回结果
        if os.path.exists(output_dir):
            print(f"Loading statistics from existing file: {output_dir}")
            return load_config(output_dir)

        # 计算均值和标准差
        print(f"Computing statistics for images in directory: {root_dir}")
        statistics = self._compute_mean_std(root_dir)
        print(statistics)

        # 保存计算结果到文件
        dump_config(statistics, output_dir)
        print(f"Statistics saved to: {output_dir}")

        return statistics
'''
class ImgStats:
    """
    用于计算指定目录下所有图像的均值和标准差，并存储到 YAML 文件中。
    """

    def __init__(self, num_threads=4):
        """
        初始化类。

        Args:
            num_threads (int): 用于并行处理图像的线程数量，默认 4。
        """
        self.num_threads = num_threads

    def _process_image(self, image_path):
        """
        处理单张图像，计算像素总和和平方和。

        Args:
            image_path (str): 图像路径。

        Returns:
            tuple: 包含像素总和、像素平方和以及图像像素数。
        """
        image = Image.open(image_path).convert("RGB")  # 确保图像是 RGB 格式
        image_array = np.array(image) / 255.0  # 归一化到 [0, 1] 范围
        pixel_sum = image_array.sum(axis=(0, 1))
        pixel_squared_sum = (image_array ** 2).sum(axis=(0, 1))
        total_pixels = image_array.shape[0] * image_array.shape[1]
        return pixel_sum, pixel_squared_sum, total_pixels

    def _compute_mean_std(self, root_dir):
        """
        遍历目录下所有图像，计算均值和标准差。

        Args:
            root_dir (str): 图像目录的路径。

        Returns:
            dict: 包含整体均值和标准差的字典：
                  {
                      "mean": (mean_r, mean_g, mean_b),
                      "std": (std_r, std_g, std_b)
                  }
        """
        pixel_sum = np.zeros(3)  # 用于累加 R/G/B 通道的像素值
        pixel_squared_sum = np.zeros(3)  # 用于累加 R/G/B 通道的像素平方值
        total_pixels = 0  # 总像素数

        # 收集所有图像路径
        image_paths = []
        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}  # 支持的扩展名
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in supported_extensions:
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            raise ValueError(f"No supported images found in the directory: {root_dir}")

        # 并行处理图像
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(tqdm(
                executor.map(self._process_image, image_paths),
                total=len(image_paths),
                desc="Processing images"
            ))

        # 聚合结果
        for pixel_sum_part, pixel_squared_sum_part, total_pixels_part in results:
            pixel_sum += pixel_sum_part
            pixel_squared_sum += pixel_squared_sum_part
            total_pixels += total_pixels_part

        # 计算均值和标准差
        mean = (pixel_sum / total_pixels).round(3).tolist()  # 保留三位小数并转为 Python 列表
        std = (np.sqrt(pixel_squared_sum / total_pixels - (pixel_sum / total_pixels) ** 2)).round(3).tolist()

        return {"mean": tuple(mean), "std": tuple(std)}

    def compute_mean_std(self, root_dir,output_dir):
        """
        计算图像均值和标准差，存储到 root_dir 中的 statistics.yaml 文件。如果文件已存在，则直接加载文件数据。

        Args:
            root_dir (str): 图像目录的路径。

        Returns:
            dict: 包含均值和标准差的字典。
        """

        # 如果文件已存在，直接加载并返回结果
        if os.path.exists(output_dir):
            print(f"Loading statistics from existing file: {output_dir}")
            return load_config(output_dir)

        # 计算均值和标准差
        print(f"Computing statistics for images in directory: {root_dir}")
        statistics = self._compute_mean_std(root_dir)

        # 保存计算结果到文件
        dump_config(statistics, output_dir)
        print(f"Statistics saved to: {output_dir}")

        return statistics
'''

