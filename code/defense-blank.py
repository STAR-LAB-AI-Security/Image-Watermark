def defense(sample: dict) -> dict:
    """水印嵌入（学生自选算法：DCT/扩频/LSB/深度隐写……）。

    参数
    ----
    sample : {"sample_id": str, "image": RGB uint8 H×W×3}

    返回
    ----
    {"image": RGB uint8 H×W×3，与输入同尺寸同 dtype}
    """
    pass


def detect(request: dict) -> dict:
    """水印检测（学生自选算法，须与 defense 配套）。

    参数
    ----
    request : {"image": RGB uint8 H×W×3（可能已被攻击处理）}

    返回
    ----
    {"watermark_probability": float 0.0~1.0}
    """
    pass
