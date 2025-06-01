def generate_custom_yolov8_yaml(nc, stages, channel_sizes, c2f_repeats, include_sppf=True):
    assert 3 <= stages <= 5, "Only 3 to 5 stages (P3 to P5) are supported."
    stages = stages-1
    backbone = []
    head = []

    # Initial Conv (P1/2 and P2/4)
    backbone.append(f"- [-1, 1, Conv, [{channel_sizes[0]}, 3, 2]]")  # 0
    backbone.append(f"- [-1, 1, Conv, [{channel_sizes[1]}, 3, 2]]")  # 1
    backbone.append(f"- [-1, {c2f_repeats[0]}, C2f, [{channel_sizes[1]}, True]]")  # 2

    last_idx = 2

    for i in range(last_idx, stages + 1):  # P3 to P5
        conv = f"- [-1, 1, Conv, [{channel_sizes[i]}, 3, 2]]"
        c2f = f"- [-1, {c2f_repeats[i-1]}, C2f, [{channel_sizes[i]}, True]]"
        backbone.append(conv)
        backbone.append(c2f)

    if include_sppf:
        backbone.append(f"- [-1, 1, SPPF, [{channel_sizes[-1]}, 5]]")

    # Calculate indices of C2f outputs
    base_indices = [2]  # C2f after P2/4
    for i in range(stages-1):
        base_indices.append(base_indices[-1] + 2)  # Each stage adds 2 layers (Conv + C2f)

    if include_sppf:
        sppf_idx = base_indices[-1] + 1
        base_indices.append(sppf_idx)  # SPPF replaces last C2f as final output

    if stages == 4:
        head.extend([
                "- [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]",                     # upsample P5
                f"- [[-1, {base_indices[2]}], 1, Concat, [1]]",                      # concat with P4
                f"- [-1, 3, C2f, [{channel_sizes[3]}]]",
                
                "- [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]",                    # upsample P4
                f"- [[-1, {base_indices[1]}], 1, Concat, [1]]",                      # concat with P3
                f"- [-1, 3, C2f, [{channel_sizes[2]}]]",
                
                f"- [-1, 1, Conv, [{channel_sizes[2]}, 3, 2]]",
                f"- [[-1,{base_indices[-1]+3}], 1, Concat, [1]]",
                f"- [-1, 3, C2f, [{channel_sizes[3]}]]",
                
                f"- [-1, 1, Conv, [{channel_sizes[3]}, 3, 2]]",
                f"- [[-1, {base_indices[-1]}], 1, Concat, [1]]",
                f"- [-1, 3, C2f, [{channel_sizes[4]}]]",
            ])
        if include_sppf:
            head.append(f"- [[15, 18, 21], 1, Detect, [nc]]")
        else:
            head.append(f"- [[14, 17,20], 1, Detect, [nc]]")

    elif stages == 3:
        head.extend([
                "- [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]",
                f"- [[-1, {base_indices[1]}], 1, Concat, [1]]",
                f"- [-1, 3, C2f, [{channel_sizes[2]}]]",
                f"- [-1, 1, Conv, [{channel_sizes[2]}, 3, 2]]",
                f"- [[-1, {base_indices[2]}], 1, Concat, [1]]",
                f"- [-1, 3, C2f, [{channel_sizes[3]}]]",
            ])
        if include_sppf:
            head.append(f"- [[10, 13], 1, Detect, [nc]]")
        else:
            head.append(f"- [[9, 12], 1, Detect, [nc]]")
    elif stages == 2:
        head.append(f"- [[{base_indices[-1]}], 1, Detect, [nc]]")

    # Final YAML string
    yaml_str = f"""# Custom YOLOv8 Model
# Auto-generated

nc: {nc}

scales:
  n: [0.33, 0.25, {max(channel_sizes)}]
  s: [0.33, 0.50, {max(channel_sizes)}]
  m: [0.67, 0.75, {max(channel_sizes)*3//4}]
  l: [1.00, 1.00, {max(channel_sizes)//2}]
  x: [1.00, 1.25, {max(channel_sizes)//2}]

backbone:
{chr(10).join(backbone)}

head:
{chr(10).join(head)}
"""

    
    return yaml_str
